/// Execute all nine criteria in record-only mode. No decision from the return
/// value is applied to export; callers decide whether to block in a later task.
pub(crate) fn run_quality_gate(
    input: &QualityGateInput<'_>,
    load_source: &mut SourceLoader<'_>,
) -> QualityGateReport {
    let mut report = QualityGateReport::default();
    let mut timing = QualityGateTimingRecord::default();
    if input.output.dimensions() != input.coverage.dimensions()
        || input.ownership.len() != input.coverage.as_raw().len()
        || input.confidence.len() != input.coverage.as_raw().len()
    {
        report.verdict = QualityGateVerdict::InsufficientEvidence;
        report.unmeasurable.push(UnmeasurableRecord {
            criterion: "all".to_string(),
            world: WorldPoint {
                x: input.world_origin.0,
                y: input.world_origin.1,
            },
            reason: "diagnostics_roi_invalid".to_string(),

            category: classify_unmeasurable_reason("diagnostics_roi_invalid"),
        });
        return report;
    }
    let large_plane =
        (input.coverage.width() as u64).saturating_mul(input.coverage.height() as u64) > 40_000_000;
    let (regions, rois) = if large_plane {
        (
            None,
            select_sparse_rois(input.coverage, input.ownership, input.world_origin),
        )
    } else {
        let regions = match quality_gate::label_owner_regions(input.ownership, input.coverage) {
            Ok(value) => value,
            Err(reason) => {
                report.verdict = QualityGateVerdict::InsufficientEvidence;
                report.unmeasurable.push(UnmeasurableRecord {
                    criterion: "all".to_string(),
                    world: WorldPoint {
                        x: input.world_origin.0,
                        y: input.world_origin.1,
                    },
                    reason: reason.to_string(),

                    category: classify_unmeasurable_reason(reason),
                });
                return report;
            }
        };
        let rois = match quality_gate::select_quality_rois(&regions, input.world_origin) {
            Ok(value) => value.rois,
            Err(reason) => {
                report.verdict = QualityGateVerdict::InsufficientEvidence;
                report.unmeasurable.push(UnmeasurableRecord {
                    criterion: "all".to_string(),
                    world: WorldPoint {
                        x: input.world_origin.0,
                        y: input.world_origin.1,
                    },
                    reason: reason.to_string(),

                    category: classify_unmeasurable_reason(reason),
                });
                return report;
            }
        };
        (Some(regions), rois)
    };
    report.roi_count = rois.len();

    let mut local_scale = Criterion::new(
        "local_scale_median",
        quality_gate::QUALITY_LOCAL_SCALE_MEDIAN_MIN,
    );
    local_scale.diagnostic = input.acceptance_render_scale < 1.0;
    let mut local_ratio = Criterion::new(
        "local_scale_pixel_ratio",
        quality_gate::QUALITY_LOCAL_SCALE_PIXEL_RATIO_MIN,
    );
    let mut mtf = Criterion::new("mtf50_normalized", quality_gate::QUALITY_MTF50_RATIO_MIN);
    let mut gradient = Criterion::new(
        "gradient_energy_normalized",
        quality_gate::QUALITY_GRADIENT_RATIO_MIN,
    );
    let mut noise = Criterion::bounded(
        "noise_sigma_ratio",
        quality_gate::QUALITY_NOISE_RATIO_MIN,
        quality_gate::QUALITY_NOISE_RATIO_MAX,
    );
    noise.diagnostic = input.final_sharpen_amount > 0.0;
    let mut delta = Criterion::new("roi_delta_e00", quality_gate::QUALITY_ROI_DELTA_E00_MAX);
    let mut unmeasurable = Vec::new();

    // Grouping keeps a decoded Source_RAW alive only for that owner's ROIs.
    let mut by_owner: BTreeMap<u16, Vec<&QualityRoi>> = BTreeMap::new();
    for roi in &rois {
        by_owner.entry(roi.owner).or_default().push(roi);
    }
    let source_map: BTreeMap<u16, &QualitySource> =
        input.sources.iter().map(|s| (s.owner, s)).collect();
    for (owner, owner_rois) in by_owner {
        let Some(source_info) = source_map.get(&owner).copied() else {
            for roi in owner_rois {
                for criterion in [
                    &mut local_scale,
                    &mut local_ratio,
                    &mut mtf,
                    &mut gradient,
                    &mut noise,
                    &mut delta,
                ] {
                    record_unmeasurable(
                        criterion,
                        &mut unmeasurable,
                        roi,
                        "owner_source_undecodable",
                    );
                }
            }
            continue;
        };
        let decode_started = Instant::now();
        let source = match load_source(source_info) {
            Ok(value) => value,
            Err(_) => {
                timing.decode_seconds += decode_started.elapsed().as_secs_f64();
                for roi in owner_rois {
                    for criterion in [
                        &mut local_scale,
                        &mut local_ratio,
                        &mut mtf,
                        &mut gradient,
                        &mut noise,
                        &mut delta,
                    ] {
                        record_unmeasurable(
                            criterion,
                            &mut unmeasurable,
                            roi,
                            "owner_source_undecodable",
                        );
                    }
                }
                continue;
            }
        };
        timing.decode_seconds += decode_started.elapsed().as_secs_f64();
        timing.owner_source_decode_count = timing.owner_source_decode_count.saturating_add(1);
        for roi in owner_rois {
            timing.roi_reference_count = timing.roi_reference_count.saturating_add(1);
            let roi_started = Instant::now();
            let where_ = world(roi);
            let output_roi = output_roi(input.output, roi);
            let reference = source_roi(
                &source,
                roi,
                &source_info.geometry,
                input.residual,
                source_info.native_registration.as_ref(),
            );
            let (Some(output_roi), Ok(reference)) = (output_roi, reference) else {
                timing.roi_sampling_seconds += roi_started.elapsed().as_secs_f64();
                for criterion in [
                    &mut local_scale,
                    &mut local_ratio,
                    &mut mtf,
                    &mut gradient,
                    &mut noise,
                    &mut delta,
                ] {
                    record_unmeasurable(
                        criterion,
                        &mut unmeasurable,
                        roi,
                        "owner_source_undecodable",
                    );
                }
                continue;
            };
            timing.reference_resampling_seconds += roi_started.elapsed().as_secs_f64();
            if let (Some(output_mean_rgb), Some(reference_mean_rgb)) = (
                quality_gate::roi_mean_rgb(&output_roi),
                quality_gate::roi_mean_rgb(&reference),
            ) {
                report.roi_photometry.push(RoiPhotometryRecord {
                    world: WorldPoint {
                        x: where_.0,
                        y: where_.1,
                    },
                    station_id: source_info.geometry.station_id,
                    owner_path: source_info.path.clone(),
                    output_mean_rgb,
                    reference_mean_rgb,
                    channel_ratio: std::array::from_fn(|c| {
                        (reference_mean_rgb[c] != 0.0).then_some(
                            f64::from(output_mean_rgb[c]) / f64::from(reference_mean_rgb[c]),
                        )
                    }),
                });
            }
            let scale_started = Instant::now();
            let scale = quality_gate::local_scale_for_roi(
                roi,
                &source_info.geometry,
                input.residual,
                input.acceptance_render_scale,
            );
            let local = scale.as_ref().ok().map(|value| value.median);
            match scale {
                Ok(value) => {
                    local_scale.push(
                        where_,
                        value.median,
                        &source_info.path,
                        value.median >= quality_gate::QUALITY_LOCAL_SCALE_MEDIAN_MIN
                            && input.acceptance_render_scale >= 1.0,
                    );
                    local_ratio.push(
                        where_,
                        value.fraction_at_least_095,
                        &source_info.path,
                        value.fraction_at_least_095
                            >= quality_gate::QUALITY_LOCAL_SCALE_PIXEL_RATIO_MIN,
                    );
                }
                Err(reason) => {
                    record_unmeasurable(&mut local_scale, &mut unmeasurable, roi, reason);
                    record_unmeasurable(&mut local_ratio, &mut unmeasurable, roi, reason);
                }
            }
            timing.local_scale_seconds += scale_started.elapsed().as_secs_f64();
            if let Some(local) = local {
                let mtf_started = Instant::now();
                match (
                    quality_gate::slanted_edge_mtf50(&output_roi, local),
                    quality_gate::slanted_edge_mtf50(&reference, 1.0),
                ) {
                    (Ok(a), Ok(b)) => {
                        if let Some((value, pass)) = ratio(
                            a.normalized,
                            b.normalized,
                            quality_gate::QUALITY_MTF50_RATIO_MIN,
                            f64::INFINITY,
                        ) {
                            mtf.push(where_, value, &source_info.path, pass);
                        } else {
                            record_unmeasurable(
                                &mut mtf,
                                &mut unmeasurable,
                                roi,
                                "mtf_reference_zero",
                            );
                        }
                    }
                    (Err(reason), _) | (_, Err(reason)) => {
                        record_unmeasurable(&mut mtf, &mut unmeasurable, roi, reason)
                    }
                }
                timing.slanted_edge_mtf_seconds += mtf_started.elapsed().as_secs_f64();
                let gradient_started = Instant::now();
                match gradient_energy_ratio(&output_roi, &reference, local) {
                    Ok(Some((value, pass))) => {
                        gradient.push(where_, value, &source_info.path, pass);
                    }
                    Ok(None) => {
                        record_unmeasurable(
                            &mut gradient,
                            &mut unmeasurable,
                            roi,
                            "gradient_reference_zero",
                        );
                    }
                    Err(reason) => {
                        record_unmeasurable(&mut gradient, &mut unmeasurable, roi, reason)
                    }
                }
                timing.gradient_seconds += gradient_started.elapsed().as_secs_f64();
            } else {
                // MTF50 and gradient energy are scale-normalised quantities;
                // substituting 1.0 would turn missing geometry into evidence.
                let reason = degradation::LOCAL_SCALE_UNMEASURABLE;
                record_unmeasurable(&mut mtf, &mut unmeasurable, roi, reason);
                record_unmeasurable(&mut gradient, &mut unmeasurable, roi, reason);
            }
            let noise_started = Instant::now();
            let full = GrayImage::from_pixel(roi.side, roi.side, image::Luma([255]));
            match (
                quality_gate::noise_sigma_ratio(&output_roi, &reference),
                quality_gate::flat_roi(&output_roi, &full),
                quality_gate::flat_roi(&reference, &full),
            ) {
                (Ok(value), Ok(_), Ok(_)) => noise.push(
                    where_,
                    value,
                    &source_info.path,
                    (quality_gate::QUALITY_NOISE_RATIO_MIN..=quality_gate::QUALITY_NOISE_RATIO_MAX)
                        .contains(&value)
                        && input.final_sharpen_amount <= 0.0,
                ),
                _ => record_unmeasurable(&mut noise, &mut unmeasurable, roi, "roi_not_flat"),
            }
            timing.noise_seconds += noise_started.elapsed().as_secs_f64();
            let delta_started = Instant::now();
            match quality_gate::roi_low_frequency_delta_e00(&output_roi, &reference) {
                Ok(value) => delta.push(
                    where_,
                    value,
                    &source_info.path,
                    value <= quality_gate::QUALITY_ROI_DELTA_E00_MAX,
                ),
                Err(reason) => record_unmeasurable(&mut delta, &mut unmeasurable, roi, reason),
            }
            timing.delta_e_seconds += delta_started.elapsed().as_secs_f64();
            timing.roi_sampling_seconds += roi_started.elapsed().as_secs_f64();
        }
    }

    let mut effective = Criterion::new(
        "effective_pixel_count",
        quality_gate::QUALITY_EFFECTIVE_PIXEL_RATIO_MIN,
    );
    let quads = input
        .sources
        .iter()
        .filter_map(source_quad)
        .collect::<Vec<_>>();
    let effective_started = Instant::now();
    let effective_result =
        effective_pixel_coverage_streaming(input.coverage, &quads, input.world_origin);
    timing.effective_coverage_seconds = effective_started.elapsed().as_secs_f64();
    match effective_result {
        Ok(EffectivePixelCoverage { ratio, .. }) => effective.push(
            input.world_origin,
            ratio,
            "",
            ratio >= quality_gate::QUALITY_EFFECTIVE_PIXEL_RATIO_MIN,
        ),
        Err(reason) => {
            effective.miss(reason);
            unmeasurable.push(UnmeasurableRecord {
                criterion: effective.name.to_string(),
                world: WorldPoint {
                    x: input.world_origin.0,
                    y: input.world_origin.1,
                },
                reason: reason.to_string(),

                category: classify_unmeasurable_reason(reason),
            });
        }
    }

    let mut boundary = Criterion::p95_with_max(
        "boundary_stroke_alignment",
        quality_gate::QUALITY_BOUNDARY_P95_MAX,
        quality_gate::QUALITY_BOUNDARY_MAX,
    );
    boundary.record_limit = 2_048;
    let boundary_started = Instant::now();
    if let Some(regions) = regions.as_ref() {
        for left in &regions.regions {
            for right in &regions.regions {
                if left.label >= right.label || left.owner == right.owner {
                    continue;
                }
                let Ok(path) = quality_gate::trace_moore_boundary(regions, left.label, right.label)
                else {
                    continue;
                };
                match quality_gate::measure_boundary_strokes(
                    input.output,
                    &path,
                    input.world_origin,
                ) {
                    Ok(measurement) => {
                        for (point, reason) in &measurement.unmeasurable {
                            boundary.miss(reason);
                            unmeasurable.push(UnmeasurableRecord {
                                criterion: boundary.name.to_string(),
                                world: WorldPoint {
                                    x: point.0,
                                    y: point.1,
                                },
                                reason: (*reason).to_string(),

                                category: classify_unmeasurable_reason(reason),
                            });
                        }
                        for value in measurement.measurements {
                            boundary.push(
                                value.world,
                                value.normal_error_px,
                                "",
                                value.normal_error_px <= quality_gate::QUALITY_BOUNDARY_MAX,
                            );
                        }
                        // The criterion is defined by both the 95th percentile
                        // and the absolute maximum. Keep the sampled points for
                        // diagnostics and add one aggregate P95 measurement so a
                        // high but sub-3px tail cannot be reported as a pass.
                        boundary.push(
                            input.world_origin,
                            measurement.p95_error_px,
                            "",
                            measurement.p95_error_px <= quality_gate::QUALITY_BOUNDARY_P95_MAX
                                && measurement.max_error_px <= quality_gate::QUALITY_BOUNDARY_MAX,
                        );
                    }
                    Err(reason) => {
                        boundary.miss(reason);
                        unmeasurable.push(UnmeasurableRecord {
                            criterion: boundary.name.to_string(),
                            world: WorldPoint {
                                x: input.world_origin.0,
                                y: input.world_origin.1,
                            },
                            reason: reason.to_string(),

                            category: classify_unmeasurable_reason(reason),
                        });
                    }
                }
            }
        }
    } else {
        measure_large_boundaries(
            input.output,
            input.coverage,
            input.ownership,
            input.world_origin,
            input.sources,
            input.accepted_station_relations,
            input.residual,
            &mut boundary,
            &mut unmeasurable,
        );
    }
    timing.boundary_seconds = boundary_started.elapsed().as_secs_f64();
    timing.boundary_sample_count = boundary.observed_count + boundary.unmeasurable;

    let owner_sharpness = build_owner_sharpness_criterion(OwnerSharpnessCriterionInput {
        coverage: input.coverage,
        confidence: input.confidence,
        textured: input.textured,
        shortfall: input.owner_shortfall,
        disagreement: input.owner_disagreement,
        unresolved_pixel_count: input.unresolved_pixel_count,
        world_origin: input.world_origin,
        unmeasurable: &mut unmeasurable,
    });

    let mut criteria = Vec::new();
    for criterion in [
        local_scale,
        local_ratio,
        effective,
        mtf,
        gradient,
        noise,
        delta,
        boundary,
        owner_sharpness,
    ] {
        criteria.push(criterion.finish(&mut unmeasurable));
    }
    let stations: std::collections::BTreeSet<_> = input
        .sources
        .iter()
        .map(|s| s.geometry.station_id)
        .collect();
    let shortfall_limit = owner_shortfall_limit();
    for station_id in stations {
        let owners: std::collections::BTreeSet<_> = input
            .sources
            .iter()
            .filter(|s| s.geometry.station_id == station_id)
            .map(|s| s.owner)
            .collect();
        let station_quads: Vec<_> = input
            .sources
            .iter()
            .filter(|s| s.geometry.station_id == station_id)
            .filter_map(source_quad)
            .collect();
        let covered = input
            .ownership
            .iter()
            .zip(input.coverage.as_raw())
            .filter(|(o, c)| **c != 0 && owners.contains(o))
            .count() as u64;
        let denominator =
            effective_pixel_coverage_streaming(input.coverage, &station_quads, input.world_origin)
                .ok()
                .map(|r| r.projected_union)
                .unwrap_or(0);
        criteria[2]
            .station_statistics
            .push(QualityStationStatistic {
                station_id,
                numerator: covered,
                denominator,
                value: (denominator > 0).then_some(covered as f64 / denominator as f64),
                ..Default::default()
            });
        if let (Some(textured), Some(shortfall)) = (input.textured, input.owner_shortfall) {
            let mut textured_count = 0u64;
            let mut flat_count = 0u64;
            let mut textured_low = 0u64;
            let mut flat_low = 0u64;
            let mut textured_shortfall = 0u64;
            let mut flat_shortfall = 0u64;
            for (index, owner) in input.ownership.iter().enumerate() {
                if input.coverage.as_raw()[index] == 0 || !owners.contains(owner) {
                    continue;
                }
                if shortfall[index] == OWNER_EVIDENCE_UNKNOWN {
                    continue;
                }
                let low = !input.confidence[index].is_finite() || input.confidence[index] < 0.05;
                let exceeds = shortfall[index] > shortfall_limit;
                if textured[index] != 0 {
                    textured_count += 1;
                    textured_low += u64::from(low);
                    textured_shortfall += u64::from(exceeds);
                } else {
                    flat_count += 1;
                    flat_low += u64::from(low);
                    flat_shortfall += u64::from(exceeds);
                }
            }
            if let Some(owner_criterion) = criteria
                .iter_mut()
                .find(|criterion| criterion.name == "owner_sharpness_coverage")
            {
                owner_criterion
                    .station_statistics
                    .push(QualityStationStatistic {
                        station_id,
                        numerator: textured_count.saturating_sub(textured_shortfall),
                        denominator: textured_count,
                        value: (textured_count > 0).then_some(
                            1.0 - textured_shortfall as f64 / textured_count.max(1) as f64,
                        ),
                        textured_pixel_count: textured_count,
                        excluded_flat_pixel_count: flat_count,
                        textured_low_confidence_count: textured_low,
                        flat_low_confidence_count: flat_low,
                        textured_low_confidence_ratio: (textured_count > 0)
                            .then_some(textured_low as f64 / textured_count as f64),
                        flat_low_confidence_ratio: (flat_count > 0)
                            .then_some(flat_low as f64 / flat_count as f64),
                        textured_shortfall_count: textured_shortfall,
                        flat_shortfall_count: flat_shortfall,
                        textured_shortfall_ratio: (textured_count > 0)
                            .then_some(textured_shortfall as f64 / textured_count as f64),
                        flat_shortfall_ratio: (flat_count > 0)
                            .then_some(flat_shortfall as f64 / flat_count as f64),
                    });
            }
        }
    }
    report.verdict = overall_verdict(&criteria);
    report.criteria = criteria;
    report.unmeasurable = unmeasurable;
    report.timing = timing;
    report
}
