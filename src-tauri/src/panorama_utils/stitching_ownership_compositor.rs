use super::*;

/// Compose already-fused focus tiles in one ownership pass.
///
/// The progressive seam compositor is appropriate for a short photographic
/// panorama, but it is order dependent: every new tile inherits the previous
/// tile's exposure estimate and seam decision.  A long scan made from virtual
/// focus tiles can therefore accumulate rectangular tone steps even when the
/// tile poses are correct.  This compositor keeps the geometry and ownership
/// decisions independent of import/path order.  In an overlap, the tile whose
/// source sample is farther from its own border owns the pixel; this favours
/// native, sharp interior pixels and prevents averaging two slightly displaced
/// focus results.
pub fn focus_tile_ownership_stitcher<R: Runtime, F>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<Rgb32FImage, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    focus_tile_ownership_stitcher_with_finishing(
        images,
        global_homographies,
        projection,
        TileCompositorFinishing::LegacyOwnership,
        app_handle,
        progress_event,
        load_image,
    )
    .map(LayeredOwnershipRender::into_image)
}

/// The default Tile_Compositor of the layered station pipeline (任务 13.1).
///
/// Geometry and ownership are decided exactly as in
/// [`focus_tile_ownership_stitcher`] — that compositor's semantics are what
/// became the default — but the finishing stage differs:
///
/// * no `crop_to_valid_rectangle()` fallback.  The canvas is the union
///   axis-aligned bounding box of the Coverage_Mask and nothing else
///   (需求 10.2 / 10.3, 任务 13.3).  Projective corners stay uncovered instead
///   of being cropped away.
/// * no unconditional final sharpen.  The default amount is 0.0 and the
///   environment variable is diagnostic only (需求 11.6 / 11.7, 任务 13.4).
///
/// Known defect on real material, to be repaired by the remaining stage-6 and
/// stage-3 tasks: source-interior ownership turns a projected tile footprint
/// into a visible polygon on a long scan.  任务 13.2 adds the
/// coverage-boundary-distance term to the seam cost and 任务 7.10 moves pixel
/// writes onto the hard Ownership_Map; until then a station boundary can be
/// visible where the two stations disagree photometrically.
#[allow(dead_code)]
pub fn layered_virtual_tile_compositor<R: Runtime, F>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<Rgb32FImage, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    layered_virtual_tile_compositor_with_ownership(
        images,
        global_homographies,
        projection,
        app_handle,
        progress_event,
        load_image,
    )
    .map(LayeredOwnershipRender::into_image)
}

/// The default layered compositor result before the public image-only API drops
/// its semantic planes. `sampling_origin` is the world/projected coordinate of
/// output pixel `(0, 0)` and therefore defines the exact Source_RAW
/// correspondence for a non-integer projective sample (需求 3.6).
#[derive(Debug, Clone)]
pub(crate) struct LayeredOwnershipRender {
    pub(crate) image: Rgb32FImage,
    pub(crate) coverage: CoverageMask,
    pub(crate) ownership: OwnershipMap,
    pub(crate) sampling_origin: (f64, f64),
    /// Test-only evidence from the production full-resolution sampling loop.
    /// This field is not present in release builds and cannot affect output.
    #[cfg(test)]
    pub(crate) full_resolution_samples: usize,
    /// Per-output-pixel sampling counts from the production loop. This is
    /// test-only evidence for checking Property 44 without aggregate masking.
    #[cfg(test)]
    pub(crate) full_resolution_sample_counts: Vec<u16>,
}

impl LayeredOwnershipRender {
    #[allow(dead_code)]
    pub(crate) fn into_image(self) -> Rgb32FImage {
        self.image
    }
}

/// Provenance-preserving form of [`layered_virtual_tile_compositor`]. The
/// image-only production entry point above calls this exact function; exposing
/// the masks prevents tests and later pipeline stages from reconstructing
/// ownership from pixel colours.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layered_virtual_tile_compositor_with_ownership<R: Runtime, F>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<LayeredOwnershipRender, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    focus_tile_ownership_stitcher_with_finishing(
        images,
        global_homographies,
        projection,
        TileCompositorFinishing::LayeredVirtualTile,
        app_handle,
        progress_event,
        load_image,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn focus_tile_ownership_stitcher_with_finishing<R: Runtime, F>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    finishing: TileCompositorFinishing,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<LayeredOwnershipRender, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    let residual_model = residual_warp::run_model_snapshot();
    if images.is_empty() {
        return Ok(LayeredOwnershipRender {
            image: Rgb32FImage::new(0, 0),
            coverage: CoverageMask::from_gray(GrayImage::new(0, 0)),
            ownership: OwnershipMap::new(0, 0, Vec::new(), Vec::new())?,
            sampling_origin: (0.0, 0.0),
            #[cfg(test)]
            full_resolution_samples: 0,
            #[cfg(test)]
            full_resolution_sample_counts: Vec::new(),
        });
    }
    let (min_x, max_x, min_y, max_y) = output_bounds(images, global_homographies, projection);
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return Ok(LayeredOwnershipRender {
            image: Rgb32FImage::new(0, 0),
            coverage: CoverageMask::from_gray(GrayImage::new(0, 0)),
            ownership: OwnershipMap::new(0, 0, Vec::new(), Vec::new())?,
            sampling_origin: (0.0, 0.0),
            #[cfg(test)]
            full_resolution_samples: 0,
            #[cfg(test)]
            full_resolution_sample_counts: Vec::new(),
        });
    }
    let (offset_x, out_width) = pixel_aligned_canvas(min_x, max_x);
    let (offset_y, out_height) = pixel_aligned_canvas(min_y, max_y);
    // 需求 10.11: an unsupported canvas is rejected before any buffer exists.
    if finishing == TileCompositorFinishing::LayeredVirtualTile
        && let Some(rejection) = super::super::stack_pipeline::compositor::reject_oversized_canvas(
            u64::from(out_width),
            u64::from(out_height),
        )
    {
        return Err(rejection);
    }
    // Only the retired comparison path reads its diagnostic switches; the
    // default layered path has no environment precondition (任务 13.7).
    let switches = ownership_compositor_switches(finishing);
    let mut panorama = Rgb32FImage::new(out_width, out_height);
    let mut panorama_mask = GrayImage::new(out_width, out_height);
    // An 8-bit interior-distance score is sufficient for deterministic tile
    // ownership and is substantially smaller than a full floating-point score
    // plane for a 40k-pixel-wide scan.
    let mut ownership = vec![0u8; out_width as usize * out_height as usize];
    // Keep the virtual-tile identity separately from the interior-distance
    // quality.  The latter decides which tile supplies detail; the former is
    // needed after composition to solve one global low-frequency tone field
    // instead of accumulating pairwise exposure steps in import order.
    let mut owner_map = GrayImage::new(out_width, out_height);
    // Source-level semantic plane for the exact same write decisions. The u16
    // identifier is not capped at the 8-bit tone-group key above.
    let mut source_owner_ids = vec![NO_OWNER; out_width as usize * out_height as usize];
    #[cfg(test)]
    let full_resolution_samples = AtomicUsize::new(0);
    #[cfg(test)]
    let full_resolution_sample_counts = (0..out_width as usize * out_height as usize)
        .map(|_| AtomicU16::new(0))
        .collect::<Vec<_>>();
    let mut tone_tiles = Vec::with_capacity(images.len());
    let mut tone_tile_ids = Vec::with_capacity(images.len());
    // Station indices and accepted Station_Relations of this run, published by
    // the station solve; without them no tile pair is tone evidence.
    let tone_topology = tone::run_topology_snapshot().unwrap_or_default();
    let row_stride = out_width as usize * 3;
    // A low-frequency consensus is accumulated from every virtual tile while
    // ownership is decided.  The final image keeps one tile's high-frequency
    // detail, while this analysis-resolution field retains the shared paper
    // illumination across overlaps and removes projective tile polygons.
    let (tone_width, tone_height) =
        focus_analysis_dimensions(out_width, out_height, out_width.max(out_height));
    let tone_stride = tone_width as usize;
    let mut tone_sum = vec![[0.0f32; 3]; tone_stride * tone_height as usize];
    let mut tone_weight = vec![0.0f32; tone_stride * tone_height as usize];
    let mut tone_count = vec![0u16; tone_stride * tone_height as usize];
    // A diagnostic/general fallback for material whose canvas tone is not
    // separable from the painted foreground.  The default semantic gate keeps
    // strokes out of the illumination solve; when disabled, the same
    // analysis-resolution consensus is still low-pass filtered before it is
    // applied, so it cannot replace the owned high-frequency detail.
    let tone_all_pixels = switches.tone_all_pixels;
    let soft_owner_boundary = switches.soft_owner_boundary;

    // Pairwise exposure estimates against the progressively built panorama
    // are order dependent.  An optional bounded preview pass solves one
    // global RGB log-gain per virtual tile from all geometric overlaps, then
    // the full-resolution pass below applies that fixed model.  It is
    // opt-in while the cost is being evaluated because a tile preview is an
    // additional decode/render of each virtual station.
    let photometric_models: Option<HashMap<usize, PhotometricModel>> = if images.len() > 1 {
        let mut previews = Vec::with_capacity(images.len());
        for image_info in images {
            let image = load_image(image_info)?;
            let longest = image.width().max(image.height()).max(1);
            let scale = (1536.0 / longest as f64).min(1.0);
            let preview_width = ((image.width() as f64 * scale).round() as u32).max(2);
            let preview_height = ((image.height() as f64 * scale).round() as u32).max(2);
            let preview = resize_rgb(&image, preview_width, preview_height);
            let source_scale = Matrix3::new(
                image.width() as f64 / preview_width as f64,
                0.0,
                0.0,
                0.0,
                image.height() as f64 / preview_height as f64,
                0.0,
                0.0,
                0.0,
                1.0,
            );
            let transform = global_homographies
                .get(&image_info.id)
                .copied()
                .ok_or_else(|| format!("Missing focus tile pose for '{}'", image_info.filename))?
                * source_scale;
            previews.push((preview, transform));
        }
        let calibration = calibrate_overlap_photometry(&previews, &PhotometricOptions::default());
        let reliable_pairs = calibration
            .pairs
            .iter()
            .filter(|pair| pair.reliable)
            .count();
        println!(
            "  - Global virtual-tile photometry: reliable_pairs={}/{} error={:.4}->{:.4}",
            reliable_pairs,
            calibration.pairs.len(),
            calibration.held_out_constant_error,
            calibration.held_out_corrected_error
        );
        // Observation only: unreliable pairs already fall back to the
        // identity model inside `calibrate_overlap_photometry`.
        if reliable_pairs < calibration.pairs.len() {
            degradation::record_run_degradation(
                degradation::TONE_INSUFFICIENT_SAMPLES,
                serde_json::json!({
                    "stage": "global_virtual_tile_photometry",
                    "reliable_pairs": reliable_pairs,
                    "pairs": calibration.pairs.len(),
                }),
            );
        }
        Some(
            images
                .iter()
                .enumerate()
                .map(|(index, image)| (image.id, calibration.models[index].clone()))
                .collect(),
        )
    } else {
        None
    };

    for (index, image_info) in images.iter().enumerate() {
        let _ = app_handle.emit(
            progress_event,
            format!("Composing focus tile {} of {}", index + 1, images.len()),
        );
        println!(
            "  - Ownership composing '{}', tile {} of {}",
            image_info.filename,
            index + 1,
            images.len()
        );
        let homography = global_homographies
            .get(&image_info.id)
            .ok_or_else(|| format!("Missing focus tile pose for '{}'", image_info.filename))?;
        let inverse = homography.try_inverse().ok_or_else(|| {
            format!(
                "Focus tile pose for '{}' is not invertible",
                image_info.filename
            )
        })?;
        let tile = load_image(image_info)?;
        let tile_width = tile.width().max(1) as f64;
        let tile_height = tile.height().max(1) as f64;
        let tile_short = tile_width.min(tile_height).max(1.0);
        // The layered compositor keeps hard Source_RAW ownership exact.  The
        // calibration is still solved unconditionally for the Tone_Harmonizer
        // report, while the retired comparison path is the only one that
        // applies its pixels during this legacy ownership pass.
        let photo_model = (finishing == TileCompositorFinishing::LegacyOwnership)
            .then(|| {
                photometric_models
                    .as_ref()
                    .and_then(|models| models.get(&image_info.id))
            })
            .flatten();
        // Requirement 3.6: the default ownership path may choose a sample, but
        // it must not change that sample's RGB while continuing to name the
        // source as owner. Keep exposure compensation only on the retired
        // comparison path.
        let exposure_enabled = finishing == TileCompositorFinishing::LegacyOwnership
            && photo_model.is_none()
            && !switches.skip_exposure_gain;
        let exposure = if !exposure_enabled {
            ExposureCompensation {
                cell_size: 1,
                grid_width: 1,
                grid_height: 1,
                gains: vec![1.0],
                representative_gain: 1.0,
                channel_gains: vec![[1.0; 3]],
                representative_channel_gain: [1.0; 3],
            }
        } else {
            estimate_overlap_exposure_compensation(ExposureOverlap {
                panorama: &panorama,
                panorama_mask: &panorama_mask,
                candidate: image_info,
                candidate_image: &tile,
                candidate_inverse: &inverse,
                projection,
                offset_x,
                offset_y,
            })
        };
        // Sample the candidate at the bounded tone-analysis resolution. Only
        // canvas-like pixels contribute; brush strokes and seals cannot steer
        // the exposure field. A raised-cosine edge weight favours native tile
        // interiors and makes the consensus independent of tile order.
        for tone_y in 0..tone_height {
            let canvas_y = (tone_y as f64 + 0.5) * out_height as f64 / tone_height.max(1) as f64;
            for tone_x in 0..tone_width {
                let canvas_x = (tone_x as f64 + 0.5) * out_width as f64 / tone_width.max(1) as f64;
                let target = Point3::new(canvas_x - offset_x, canvas_y - offset_y, 1.0);
                let Some(source) = map_target_to_source_with_residual(
                    &inverse,
                    target,
                    image_info,
                    projection,
                    &residual_model,
                ) else {
                    continue;
                };
                if source.x < 0.0
                    || source.y < 0.0
                    || source.x >= tile_width
                    || source.y >= tile_height
                {
                    continue;
                }
                let mut pixel = get_high_quality_interpolated_pixel(&tile, source.x, source.y);
                if let Some(model) = photo_model {
                    let preview_x = source.x * model.preview_width as f64 / tile_width;
                    let preview_y = source.y * model.preview_height as f64 / tile_height;
                    pixel = model.apply(pixel, preview_x, preview_y);
                }
                if !tone_all_pixels
                    && (!focus_stack_pixel_is_canvas_like(&pixel.0)
                        || focus_stack_pixel_is_tone_foreground(&pixel.0))
                {
                    continue;
                }
                let edge_distance = source
                    .x
                    .min(source.y)
                    .min((tile_width - 1.0) - source.x)
                    .min((tile_height - 1.0) - source.y)
                    .max(0.0);
                let edge_t = (edge_distance / (tile_short * 0.20).max(1.0)).clamp(0.0, 1.0);
                let weight = (edge_t * edge_t * (3.0 - 2.0 * edge_t)) as f32;
                if weight <= 0.01 {
                    continue;
                }
                let tone_index = tone_y as usize * tone_stride + tone_x as usize;
                for channel in 0..3 {
                    tone_sum[tone_index][channel] += pixel[channel] * weight;
                }
                tone_weight[tone_index] += weight;
                tone_count[tone_index] = tone_count[tone_index].saturating_add(1);
            }
        }
        // Do not estimate a gain against the progressively accumulated
        // panorama here.  That would make this supposedly order-independent
        // ownership pass inherit a chain of pairwise exposure decisions: a
        // tile selected later could be corrected against a previous owner's
        // already corrected tone, then replaced by a third tile whose gain was
        // measured against a different surface.  This is the source of broad
        // tile-sized tone bands on long scans.  Ownership depends only on the
        // source geometry/edge distance; the owner-wide harmonization below
        // solves the low-frequency tone once, after all owners are final.
        let (left, right, top, bottom) = transformed_image_region(
            image_info, homography, projection, offset_x, offset_y, out_width, out_height,
        )
        .unwrap_or((
            0,
            out_width.saturating_sub(1),
            0,
            out_height.saturating_sub(1),
        ));
        if left > right || top > bottom {
            continue;
        }
        let left = left.min(out_width.saturating_sub(1));
        let right = right.min(out_width.saturating_sub(1));
        let top = top.min(out_height.saturating_sub(1));
        let bottom = bottom.min(out_height.saturating_sub(1));
        // The tone field reads covered tile pixels only and is resampled
        // through the same world-to-tile map the pixels below are drawn with,
        // so neighbouring tiles are compared where they show the same content.
        tone_tile_ids.push(image_info.id);
        tone_tiles.push(tone::world_aligned_tone_tile(
            tone_topology
                .station_of(image_info.id)
                .unwrap_or(image_info.id),
            (index + 1).min(u16::MAX as usize) as u16,
            &tile,
            (left, top, right, bottom),
            |x, y| {
                map_target_to_source_with_residual(
                    &inverse,
                    Point3::new(x - offset_x, y - offset_y, 1.0),
                    image_info,
                    projection,
                    &residual_model,
                )
                .map(|source| (source.x, source.y))
            },
        ));
        let image_width = tile.width() as f64;
        let image_height = tile.height() as f64;
        panorama
            .as_mut()
            .par_chunks_mut(row_stride)
            .zip(ownership.par_chunks_mut(out_width as usize))
            .zip(panorama_mask.as_mut().par_chunks_mut(out_width as usize))
            .zip(owner_map.as_mut().par_chunks_mut(out_width as usize))
            .zip(source_owner_ids.par_chunks_mut(out_width as usize))
            .enumerate()
            .skip(top as usize)
            .take((bottom - top + 1) as usize)
            .for_each(
                |(y, ((((row, quality_row), mask_row), owner_row), source_owner_row))| {
                    for x in left..=right {
                        let target = Point3::new(x as f64 - offset_x, y as f64 - offset_y, 1.0);
                        let Some(source) = map_target_to_source_with_residual(
                            &inverse,
                            target,
                            image_info,
                            projection,
                            &residual_model,
                        ) else {
                            continue;
                        };
                        if source.x < 0.0
                            || source.y < 0.0
                            || source.x >= image_width
                            || source.y >= image_height
                        {
                            continue;
                        }
                        let edge_distance = source
                            .x
                            .min(source.y)
                            .min((image_width - 1.0) - source.x)
                            .min((image_height - 1.0) - source.y)
                            .max(0.0);
                        let quality = (1.0 + (edge_distance / tile_short * 510.0).round())
                            .clamp(1.0, 255.0) as u8;
                        let quality_slot = &mut quality_row[x as usize];
                        let candidate_owner = (index + 1).min(255) as u8;
                        if !focus_tile_ownership_should_replace(
                            *quality_slot,
                            owner_row[x as usize],
                            quality,
                            candidate_owner,
                        ) {
                            continue;
                        }
                        #[cfg(test)]
                        full_resolution_sample_counts[y * out_width as usize + x as usize]
                            .fetch_add(1, Ordering::Relaxed);
                        #[cfg(test)]
                        full_resolution_samples.fetch_add(1, Ordering::Relaxed);
                        let mut color =
                            get_high_quality_interpolated_pixel(&tile, source.x, source.y);
                        if let Some(model) = photo_model {
                            let preview_x = source.x * model.preview_width as f64 / image_width;
                            let preview_y = source.y * model.preview_height as f64 / image_height;
                            color = model.apply(color, preview_x, preview_y);
                        }
                        // Unfilled projective corners are transparent zeros.  Do
                        // not let them win ownership over a neighbouring tile.
                        if color[0].max(color[1]).max(color[2]) <= 1e-6 {
                            continue;
                        }
                        color = apply_exposure_compensation(color, &exposure, x, y as u32);
                        // Interior distance is a useful prior, but selecting a
                        // whole projective polygon solely by that prior cuts the
                        // paper weave at a visible hard edge. If the candidate is
                        // only marginally better and disagrees strongly with the
                        // already selected sample, keep the existing owner and
                        // let the seam remain in a locally coherent region.
                        if owner_row[x as usize] > 0 {
                            let start = x as usize * 3;
                            let current = &row[start..start + 3];
                            let current_rgb = [current[0], current[1], current[2]];
                            let current_luma =
                                current[0] * 0.299 + current[1] * 0.587 + current[2] * 0.114;
                            let candidate_luma =
                                color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114;
                            let luma_delta = (candidate_luma - current_luma).abs();
                            let quality_ratio =
                                f32::from(quality) / f32::from((*quality_slot).max(1));
                            if luma_delta > 0.075 && quality_ratio < 1.35 {
                                continue;
                            }
                            // Interior distance chooses the sharper source, but a
                            // projective tile boundary should not become a hard
                            // tone edge.  Feather only small, low-contrast canvas
                            // disagreements; saturated or high-contrast detail
                            // remains single-owner so strokes cannot double.
                            if soft_owner_boundary
                                && luma_delta < 0.06
                                && focus_stack_pixel_is_canvas_like(&current_rgb)
                                && focus_stack_pixel_is_canvas_like(&color.0)
                            {
                                let quality_advantage = (f32::from(quality)
                                    - f32::from((*quality_slot).max(1)))
                                .max(0.0);
                                let alpha =
                                    (0.45 + quality_advantage / 255.0 * 0.45).clamp(0.45, 0.90);
                                for channel in 0..3 {
                                    color[channel] = current_rgb[channel] * (1.0 - alpha)
                                        + color[channel] * alpha;
                                }
                            }
                        }
                        let start = x as usize * 3;
                        row[start..start + 3].copy_from_slice(&color.0);
                        *quality_slot = quality;
                        mask_row[x as usize] = 255;
                        owner_row[x as usize] = candidate_owner;
                        source_owner_row[x as usize] = (index + 1).min(u16::MAX as usize) as u16;
                    }
                },
            );
    }

    // Re-solve accepted inter-station overlaps only after every covered pixel
    // has a hard source owner. The local32 application changes only the strip
    // between the old and new boundaries; coverage and provenance stay exact.
    if finishing == TileCompositorFinishing::LayeredVirtualTile
        && !tone_topology.accepted.is_empty()
    {
        let (plans, planning) = build_station_seam_plans(
            images,
            global_homographies,
            projection,
            &tone_topology.accepted,
            offset_x,
            offset_y,
            out_width,
            out_height,
            &residual_model,
            &source_owner_ids,
            load_image,
        )?;
        let changed =
            apply_station_seam_plans(&plans, &mut panorama, &mut owner_map, &mut source_owner_ids)?;
        if !plans.is_empty() {
            println!(
                "  - Station seam solver: plans={}, changed_owner_pixels={}, index_owner_reads={}, transition_queries={}, endpoint_loads={}",
                plans.len(),
                changed.changed_pixels,
                planning.index_owner_reads,
                planning.relation_query_visits,
                planning.endpoint_tile_loads,
            );
        }
    }

    // Solve a low-frequency consensus from all candidate tiles. The high
    // frequency ownership below remains hard, but its broad paper tone is
    // replaced by the overlap-supported field so tile-shaped vignetting does
    // not survive as a polygonal boundary.
    let tone_mask = GrayImage::from_fn(tone_width, tone_height, |x, y| {
        let index = y as usize * tone_stride + x as usize;
        image::Luma([u8::from(tone_weight[index] > 0.05) * 255])
    });
    let tone_samples = tone_mask
        .as_raw()
        .iter()
        .filter(|&&value| value > 0)
        .count();
    if switches.low_frequency_consensus && tone_samples >= 128 {
        let tone_reference = Rgb32FImage::from_fn(tone_width, tone_height, |x, y| {
            let index = y as usize * tone_stride + x as usize;
            let weight = tone_weight[index];
            if weight > 0.05 {
                let sum = tone_sum[index];
                Rgb([sum[0] / weight, sum[1] / weight, sum[2] / weight])
            } else {
                Rgb([0.0; 3])
            }
        });
        // The illumination step is source-sized on a scan. A tiny blur only
        // feathers the ownership edge and leaves a broad diagonal tile visible;
        // use a bounded fraction of the analysis canvas so the correction is
        // continuous across one station while preserving brush-scale detail.
        let tone_radius = (tone_width.max(tone_height) as f32 * 0.05)
            .round()
            .clamp(32.0, 192.0) as usize;
        let target_low = masked_box_blur_rgb(&tone_reference, &tone_mask, tone_radius);
        let output_analysis = resize_rgb(&panorama, tone_width, tone_height);
        let output_low = masked_box_blur_rgb(&output_analysis, &tone_mask, tone_radius);
        let target_pixels = target_low.as_raw();
        let output_pixels = output_low.as_raw();
        let mut correction_pixels = vec![0.0f32; tone_width as usize * tone_height as usize * 3];
        for index in 0..tone_mask.as_raw().len() {
            if tone_mask.as_raw()[index] == 0 {
                continue;
            }
            let start = index * 3;
            for channel in 0..3 {
                correction_pixels[start + channel] = (target_pixels[start + channel]
                    - output_pixels[start + channel])
                    .clamp(-0.24, 0.24);
            }
        }
        // A correction measured only where two tiles overlap must also reach
        // the newly exposed part of that owner. Propagate a robust median per
        // owner, then keep the local overlap residual as a small supplement.
        let owner_analysis = image::imageops::resize(
            &owner_map,
            tone_width,
            tone_height,
            image::imageops::FilterType::Nearest,
        );
        let mut owner_deltas: Vec<[Vec<f32>; 3]> = (0..=u8::MAX as usize)
            .map(|_| [Vec::new(), Vec::new(), Vec::new()])
            .collect();
        for index in 0..tone_mask.as_raw().len() {
            if tone_mask.as_raw()[index] == 0 || tone_count[index] < 2 {
                continue;
            }
            let owner = owner_analysis.as_raw()[index] as usize;
            if owner == 0 {
                continue;
            }
            let start = index * 3;
            for channel in 0..3 {
                owner_deltas[owner][channel].push(correction_pixels[start + channel]);
            }
        }
        let mut owner_delta = vec![[0.0f32; 3]; u8::MAX as usize + 1];
        let mut owner_delta_valid = vec![false; u8::MAX as usize + 1];
        for owner in 1..owner_deltas.len() {
            if owner_deltas[owner][0].len() < 8 {
                continue;
            }
            for channel in 0..3 {
                owner_delta[owner][channel] = median_f32(&mut owner_deltas[owner][channel])
                    .unwrap_or(0.0)
                    .clamp(-0.24, 0.24);
            }
            owner_delta_valid[owner] = true;
        }
        for index in 0..tone_mask.as_raw().len() {
            let owner = owner_analysis.as_raw()[index] as usize;
            if !owner_delta_valid.get(owner).copied().unwrap_or(false) {
                continue;
            }
            let start = index * 3;
            for channel in 0..3 {
                correction_pixels[start + channel] =
                    owner_delta[owner][channel] * 0.75 + correction_pixels[start + channel] * 0.25;
            }
        }
        let correction_image = Rgb32FImage::from_raw(tone_width, tone_height, correction_pixels)
            .expect("focus low-frequency correction dimensions must match");
        let correction_low = masked_box_blur_rgb(&correction_image, &tone_mask, tone_radius / 2);
        let tone_weights = tone_mask
            .as_raw()
            .iter()
            .map(|&value| f32::from(value > 0))
            .collect::<Vec<_>>();
        let tone_weight_blur = box_blur_focus_map(
            &tone_weights,
            tone_width,
            tone_height,
            (tone_radius / 2).max(4),
        );
        let x_samples = linear_samples(tone_width, out_width);
        let y_samples = linear_samples(tone_height, out_height);
        let correction_pixels = correction_low.as_raw();
        let tone_weight_ref = &tone_weight_blur;
        let image_mask_ref = panorama_mask.as_raw();
        panorama
            .as_mut()
            .par_chunks_mut(out_width as usize * 3)
            .enumerate()
            .for_each(|(y, row)| {
                let y_sample = y_samples[y];
                for x in 0..out_width as usize {
                    if image_mask_ref[y * out_width as usize + x] == 0 {
                        continue;
                    }
                    let x_sample = x_samples[x];
                    let top_left = y_sample.lower * tone_stride + x_sample.lower;
                    let top_right = y_sample.lower * tone_stride + x_sample.upper;
                    let bottom_left = y_sample.upper * tone_stride + x_sample.lower;
                    let bottom_right = y_sample.upper * tone_stride + x_sample.upper;
                    let top_weight = tone_weight_ref[top_left] * (1.0 - x_sample.upper_weight)
                        + tone_weight_ref[top_right] * x_sample.upper_weight;
                    let bottom_weight = tone_weight_ref[bottom_left]
                        * (1.0 - x_sample.upper_weight)
                        + tone_weight_ref[bottom_right] * x_sample.upper_weight;
                    let weight = (top_weight * (1.0 - y_sample.upper_weight)
                        + bottom_weight * y_sample.upper_weight)
                        .clamp(0.0, 1.0);
                    if weight <= 0.001 {
                        continue;
                    }
                    let mut delta = [0.0f32; 3];
                    for channel in 0..3 {
                        let top = correction_pixels[top_left * 3 + channel]
                            * (1.0 - x_sample.upper_weight)
                            + correction_pixels[top_right * 3 + channel] * x_sample.upper_weight;
                        let bottom = correction_pixels[bottom_left * 3 + channel]
                            * (1.0 - x_sample.upper_weight)
                            + correction_pixels[bottom_right * 3 + channel] * x_sample.upper_weight;
                        let local =
                            top * (1.0 - y_sample.upper_weight) + bottom * y_sample.upper_weight;
                        delta[channel] = local * weight;
                    }
                    let start = x * 3;
                    for channel in 0..3 {
                        row[start + channel] =
                            (row[start + channel] + delta[channel]).clamp(0.0, 1.0);
                    }
                }
            });
        println!(
            "  - Low-frequency focus consensus: samples={} radius={}px",
            tone_samples, tone_radius
        );
    }

    // Detail ownership is already final at this point. Correct only the
    // remaining owner-level tone difference using selected virtual-tile
    // regions at once; doing this after the consensus pass removes residual
    // source steps without averaging or replacing brush pixels.
    if finishing == TileCompositorFinishing::LayeredVirtualTile {
        let tone_tile_index = |image_id: usize| tone_tile_ids.iter().position(|&id| id == image_id);
        let tone_relations = tone_topology
            .accepted
            .iter()
            .filter_map(|&(left, right)| Some((tone_tile_index(left)?, tone_tile_index(right)?)))
            .collect::<Vec<_>>();
        let tone_report = tone::harmonize_tone_tiles(
            &mut panorama,
            &source_owner_ids,
            out_width,
            &tone_tiles,
            &panorama_mask,
            &tone_relations,
        );
        println!(
            "  - Tone_Harmonizer: status={:?}, tiles={}, boundary_max_delta_e00={:.3}",
            tone_report.status,
            tone_report.tiles.len(),
            tone_report.boundary_delta_e.max
        );
    }
    let empty_foreground = GrayImage::new(out_width, out_height);
    if switches.owner_tone_harmonization {
        if !switches.skip_owner_tone_harmonization {
            harmonize_focus_background_tone_with_owners(
                &mut panorama,
                &panorama_mask,
                &empty_foreground,
                &owner_map,
            );
        } else {
            println!("  - Skipping final focus owner tone harmonization (diagnostic override)");
        }
    } else {
        println!("  - Skipping focus-tile owner tone harmonization (disabled by default)");
    }
    // A hard owner map keeps brush detail from averaging across a displaced
    // seam. The streaming illumination solve remains an opt-in diagnostic for
    // scenes where a broad lighting field is preferred; the owner-based solve
    // above is the default so it cannot flatten genuine artwork contrast.
    if switches.final_low_frequency_illumination {
        super::super::mosaic::smooth_streaming_low_frequency_illumination(&mut panorama);
    }
    // Tile warps and the final cubic sampling soften native edges once more.
    // Restore a bounded amount of luma detail after all tone work; the mask
    // keeps the pass out of invalid projective corners and it never averages
    // neighbouring owners.
    // The virtual-tile renderer resamples each source once for the tile and
    // once again while placing that tile on the global canvas.  A restrained
    // luma-only unsharp pass restores the native edge energy lost to those two
    // cubic samples without inventing cross-owner detail.  Keep this below the
    // per-station pass (1.35) so long scans do not acquire halos at tile seams.
    // 任务 13.4: the default layered path resolves to 0.0 here, so
    // `sharpen_focus_tile_detail` is not executed at all; the comparison path
    // keeps its 0.85 second pass.
    let final_sharpen_amount = match finishing {
        TileCompositorFinishing::LayeredVirtualTile => layered_final_sharpen_amount(),
        TileCompositorFinishing::LegacyOwnership => switches.legacy_final_sharpen_amount,
    };
    if final_sharpen_amount > 0.0 {
        sharpen_focus_tile_detail(&mut panorama, &panorama_mask, final_sharpen_amount);
    }

    let panorama_dimensions = panorama.dimensions();
    // The comparison path keeps the complete source union only when its
    // bounding box is genuinely covered and otherwise falls back to the largest
    // valid rectangle.  The default path must not: 需求 10.2 fixes the canvas at
    // the union axis-aligned bounding box of every Coverage_Mask, and 需求 10.3
    // keeps the uncovered projective corners transparent instead of cropping
    // real coverage away to hide them (任务 13.3).
    let (valid_bounds, valid_coverage) = valid_mask_bounds_and_coverage(&panorama_mask);
    let cropped = match finishing {
        TileCompositorFinishing::LayeredVirtualTile => {
            crop_to_valid_bounds(panorama, &panorama_mask)
        }
        TileCompositorFinishing::LegacyOwnership => {
            if valid_coverage >= 0.995 {
                crop_to_valid_bounds(panorama, &panorama_mask)
            } else {
                crop_to_valid_rectangle(panorama, &panorama_mask)
            }
        }
    };
    if cropped.dimensions() != panorama_dimensions {
        println!(
            "  - Cropped ownership margins: {}x{} -> {}x{} (bounds coverage {:.2}%, bbox {:?})",
            panorama_dimensions.0,
            panorama_dimensions.1,
            cropped.width(),
            cropped.height(),
            valid_coverage * 100.0,
            valid_bounds
        );
    }

    // The production default crops only to the covered union bounds, so the
    // same rectangle can be applied losslessly to all semantic planes. The
    // legacy image-only caller discards these planes.
    let (crop_x, crop_y, crop_width, crop_height) = if valid_coverage > 0.0
        && (finishing == TileCompositorFinishing::LayeredVirtualTile || valid_coverage >= 0.995)
    {
        (
            valid_bounds.0,
            valid_bounds.1,
            valid_bounds.2 - valid_bounds.0 + 1,
            valid_bounds.3 - valid_bounds.1 + 1,
        )
    } else {
        (0, 0, cropped.width(), cropped.height())
    };
    let coverage_gray = if crop_x + crop_width <= panorama_mask.width()
        && crop_y + crop_height <= panorama_mask.height()
    {
        image::imageops::crop_imm(&panorama_mask, crop_x, crop_y, crop_width, crop_height)
            .to_image()
    } else {
        GrayImage::new(cropped.width(), cropped.height())
    };
    let mut cropped_owners = vec![NO_OWNER; cropped.width() as usize * cropped.height() as usize];
    #[cfg(test)]
    let mut cropped_sample_counts =
        vec![0u16; cropped.width() as usize * cropped.height() as usize];
    if crop_x + cropped.width() <= out_width && crop_y + cropped.height() <= out_height {
        cropped_owners
            .par_chunks_mut(cropped.width() as usize)
            .enumerate()
            .for_each(|(y, row)| {
                let source_start = (crop_y as usize + y) * out_width as usize + crop_x as usize;
                row.copy_from_slice(&source_owner_ids[source_start..source_start + row.len()]);
            });
        #[cfg(test)]
        cropped_sample_counts
            .iter_mut()
            .enumerate()
            .for_each(|(index, count)| {
                let x = index % cropped.width() as usize;
                let y = index / cropped.width() as usize;
                *count = full_resolution_sample_counts
                    [(crop_y as usize + y) * out_width as usize + crop_x as usize + x]
                    .load(Ordering::Relaxed);
            });
    }
    let legend = images
        .iter()
        .map(|image| PathBuf::from(&image.filename))
        .collect::<Vec<_>>();
    Ok(LayeredOwnershipRender {
        image: cropped,
        coverage: CoverageMask::from_gray(coverage_gray),
        ownership: OwnershipMap::new(crop_width, crop_height, cropped_owners, legend)?,
        sampling_origin: (crop_x as f64 - offset_x, crop_y as f64 - offset_y),
        #[cfg(test)]
        full_resolution_samples: full_resolution_samples.load(Ordering::Relaxed),
        #[cfg(test)]
        full_resolution_sample_counts: cropped_sample_counts,
    })
}
