//! The Quality_Gate run over the final output.
//!
//! One function by design for now: ROI selection, source sampling, the nine
//! criteria and the report run in a fixed order.

use super::super::degradation;
use super::super::quality_gate;
use super::super::quality_gate::EffectivePixelCoverage;
use super::super::quality_gate::QualityRoi;
use super::super::report::QualityGateReport;
use super::super::report::QualityGateTimingRecord;
use super::super::report::QualityGateVerdict;
use super::super::report::QualityStationStatistic;
use super::super::report::RoiPhotometryRecord;
use super::super::report::UnmeasurableRecord;
use super::super::report::WorldPoint;
use image::GrayImage;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use super::{
    Criterion, OWNER_EVIDENCE_UNKNOWN, OwnerSharpnessCriterionInput, QualityGateInput,
    QualityRoiCache, QualitySource, SourceLoader, apply_run_tone_transform,
    build_owner_sharpness_criterion, classify_unmeasurable_reason,
    effective_pixel_coverage_streaming, gradient_energy_ratio, measure_large_boundaries,
    measure_source_local_scale, measure_station_local_scale, output_roi, overall_verdict,
    owner_shortfall_limit, ratio, record_unmeasurable, select_sparse_rois,
    select_sparse_rois_with_evidence_side, source_quad, source_roi_for_quality,
    source_roi_for_station_quality, world,
};
#[cfg(test)]
use super::{
    write_effective_coverage_diagnostics, write_gradient_source_oracle,
    write_tone_owner_fidelity_oracle,
};

#[cfg(test)]
#[path = "source_visual_capture.rs"]
mod source_visual_capture;

/// Execute all nine criteria in record-only mode. No decision from the return
/// value is applied to export; callers decide whether to block in a later task.
pub(crate) fn run_quality_gate(
    input: &QualityGateInput<'_>,
    load_source: &mut SourceLoader<'_>,
) -> QualityGateReport {
    let mut cache = QualityRoiCache::new(0);
    run_quality_gate_with_cache(input, load_source, &mut cache)
}

pub(crate) fn run_quality_gate_with_cache(
    input: &QualityGateInput<'_>,
    load_source: &mut SourceLoader<'_>,
    roi_cache: &mut QualityRoiCache,
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
    #[cfg(test)]
    let mut source_visual_capture = source_visual_capture::Capture::from_environment(input);
    roi_cache.bind_input(input);
    // The small-plane region tracer allocates and walks every owner contour.
    // A 9-source scale=0.5 acceptance canvas is ~22M pixels, large enough
    // for that path to dominate the gate even though it is below the former
    // 40M cutoff.  Use the streaming selector/boundary path from 20M onward;
    // metric thresholds and ROI evidence remain unchanged.
    let large_plane =
        (input.coverage.width() as u64).saturating_mul(input.coverage.height() as u64) > 20_000_000;
    let station_evidence = station_evidence_owners(input.ownership, input.sources);
    let raw_regions = if large_plane {
        None
    } else {
        quality_gate::label_owner_regions(input.ownership, input.coverage).ok()
    };
    let raw_rois = if large_plane {
        select_sparse_rois(input.coverage, input.ownership, input.world_origin)
    } else {
        raw_regions
            .as_ref()
            .and_then(|regions| quality_gate::select_quality_rois(regions, input.world_origin).ok())
            .map(|selection| selection.rois)
            .unwrap_or_default()
    };
    let station_rois = select_station_level_rois(input, large_plane, &station_evidence);
    // Station evidence is the authoritative geometry contract for a focus
    // stack.  Prefer it whenever it reaches the gate's measurable minimum;
    // comparing only list lengths lets a dense raw-owner lattice (224 ROIs in
    // the real 9-source run) silently win and brings back the fragmented
    // owner provenance that this path is meant to remove.
    let (regions, rois) = if station_rois.len() >= quality_gate::QUALITY_MIN_MEASURABLE {
        (raw_regions, station_rois)
    } else {
        (raw_regions, raw_rois)
    };
    if rois.is_empty() {
        report.verdict = QualityGateVerdict::InsufficientEvidence;
        report.unmeasurable.push(UnmeasurableRecord {
            criterion: "all".to_string(),
            world: WorldPoint {
                x: input.world_origin.0,
                y: input.world_origin.1,
            },
            reason: degradation::QUALITY_GATE_INSUFFICIENT_EVIDENCE.to_string(),
            category: classify_unmeasurable_reason(degradation::QUALITY_GATE_INSUFFICIENT_EVIDENCE),
        });
        #[cfg(test)]
        source_visual_capture::finish(&mut source_visual_capture, &report);
        return report;
    }
    report.roi_count = rois.len();
    timing.station_owner_evidence_roi_count = rois
        .iter()
        .filter(|roi| quality_gate::station_level_roi_station_id(roi.region_label).is_some())
        .count();
    timing.raw_owner_evidence_roi_count = rois
        .len()
        .saturating_sub(timing.station_owner_evidence_roi_count);
    #[cfg(test)]
    if let Some(capture) = source_visual_capture.as_mut() {
        capture.select_rois(&rois, input.sources);
    }
    // Large acceptance planes use station evidence windows. Apply the
    // criterion-specific content prefilters only there; small fixtures retain
    // the historical per-ROI unmeasurable records and thresholds.
    let criterion_specific_rois = large_plane && timing.station_owner_evidence_roi_count > 0;

    let mut local_scale = Criterion::new(
        "local_scale_median",
        quality_gate::QUALITY_LOCAL_SCALE_MEDIAN_MIN,
    );
    local_scale.diagnostic = input.acceptance_render_scale < 1.0;
    let mut local_ratio = Criterion::new(
        "local_scale_pixel_ratio",
        quality_gate::QUALITY_LOCAL_SCALE_PIXEL_RATIO_MIN,
    );
    // Source and output are both scaled by the diagnostic render path.
    // Their relative Jacobian still exposes geometric loss, so keep this
    // gate active even when the acceptance render is below full resolution.
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
    let mut station_source_images = BTreeMap::new();
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
                    if criterion_specific_rois
                        && matches!(criterion.name, "mtf50_normalized" | "noise_sigma_ratio")
                    {
                        continue;
                    }
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
        let mut cached_references = owner_rois
            .iter()
            .map(|roi| roi_cache.take(source_info, roi))
            .collect::<Vec<_>>();
        let source = if cached_references.iter().any(Option::is_none) {
            let decode_started = Instant::now();
            let source = load_source(source_info).ok();
            timing.decode_seconds += decode_started.elapsed().as_secs_f64();
            if source.is_some() {
                timing.owner_source_decode_count =
                    timing.owner_source_decode_count.saturating_add(1);
            }
            source
        } else {
            None
        };
        for (roi, cached_reference) in owner_rois.into_iter().zip(cached_references.drain(..)) {
            #[cfg(test)]
            eprintln!(
                "  - Quality_Gate ROI start: owner={} station={:?} x={} y={} side={} cached={}",
                roi.owner,
                quality_gate::station_level_roi_station_id(roi.region_label),
                roi.x,
                roi.y,
                roi.side,
                cached_reference.is_some()
            );
            timing.roi_reference_count = timing.roi_reference_count.saturating_add(1);
            let roi_started = Instant::now();
            let where_ = world(roi);
            let output_roi = output_roi(input.output, roi);
            let was_cached = cached_reference.is_some();
            let station_level = quality_gate::station_level_roi_station_id(roi.region_label);
            let mut reference = match cached_reference {
                Some(reference) => Ok(reference),
                None => source
                    .as_ref()
                    .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)
                    .and_then(|source| {
                        if let Some(station_id) = station_level {
                            source_roi_for_station_quality(
                                &mut station_source_images,
                                owner,
                                Some(source),
                                input.ownership,
                                input.coverage.width(),
                                roi,
                                station_id,
                                &source_map,
                                input.residual,
                                load_source,
                            )
                        } else {
                            source_roi_for_quality(source, roi, source_info, input.residual)
                        }
                    }),
            };
            let (Some(output_roi), Ok(reference)) = (output_roi, reference.as_mut()) else {
                timing.roi_sampling_seconds += roi_started.elapsed().as_secs_f64();
                for criterion in [
                    &mut local_scale,
                    &mut local_ratio,
                    &mut mtf,
                    &mut gradient,
                    &mut noise,
                    &mut delta,
                ] {
                    if criterion_specific_rois
                        && matches!(criterion.name, "mtf50_normalized" | "noise_sigma_ratio")
                    {
                        continue;
                    }
                    record_unmeasurable(
                        criterion,
                        &mut unmeasurable,
                        roi,
                        "owner_source_undecodable",
                    );
                }
                continue;
            };
            if !was_cached && source_info.native_registration.is_none() {
                apply_run_tone_transform(reference, source_info.geometry.station_id);
            }
            let reference = reference.clone();
            timing.reference_resampling_seconds += roi_started.elapsed().as_secs_f64();
            #[cfg(test)]
            if let Some(capture) = source_visual_capture.as_mut() {
                capture.record(input, roi, &output_roi, &reference, was_cached);
            }
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
            let scale = if let Some(station_id) = station_level {
                measure_station_local_scale(
                    roi,
                    input.ownership,
                    input.coverage.width(),
                    station_id,
                    &source_map,
                    input.residual,
                    input.acceptance_render_scale,
                )
            } else {
                measure_source_local_scale(
                    roi,
                    source_info,
                    input.residual,
                    input.acceptance_render_scale,
                )
            };
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
                if !criterion_specific_rois {
                    match (
                        quality_gate::slanted_edge_mtf50(&output_roi, local),
                        // source_roi has already sampled the original onto this
                        // same world lattice. Both measurements therefore use
                        // the same native-to-world pixel scale; treating this
                        // reference as an unwarped native crop normalises twice.
                        quality_gate::slanted_edge_mtf50(&reference, local),
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
                }
                timing.slanted_edge_mtf_seconds += mtf_started.elapsed().as_secs_f64();
                let gradient_started = Instant::now();
                match gradient_energy_ratio(&output_roi, &reference, local) {
                    Ok(Some((value, pass))) => {
                        #[cfg(test)]
                        crate::panorama_stitching::acceptance_tests::capture_native_gradient_roi(
                            where_,
                            &source_info.path,
                            value,
                            local,
                            &output_roi,
                            &reference,
                        );
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
                if !criterion_specific_rois {
                    record_unmeasurable(&mut mtf, &mut unmeasurable, roi, reason);
                }
                record_unmeasurable(&mut gradient, &mut unmeasurable, roi, reason);
            }
            let noise_started = Instant::now();
            if !criterion_specific_rois {
                let full = GrayImage::from_pixel(roi.side, roi.side, image::Luma([255]));
                match (
                    quality_gate::flat_roi(&output_roi, &full),
                    quality_gate::flat_roi(&reference, &full),
                    quality_gate::noise_sigma_ratio(&output_roi, &reference),
                ) {
                    (Ok(_), Ok(_), Ok(value)) => noise.push(
                        where_,
                        value,
                        &source_info.path,
                        (quality_gate::QUALITY_NOISE_RATIO_MIN
                            ..=quality_gate::QUALITY_NOISE_RATIO_MAX)
                            .contains(&value)
                            && input.final_sharpen_amount <= 0.0,
                    ),
                    (Err(reason), _, _) | (_, Err(reason), _) | (_, _, Err(reason)) => {
                        record_unmeasurable(&mut noise, &mut unmeasurable, roi, reason)
                    }
                }
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
            roi_cache.insert(source_info, roi, reference);
            #[cfg(test)]
            eprintln!(
                "  - Quality_Gate ROI done: owner={} elapsed={:.2}s",
                roi.owner,
                roi_started.elapsed().as_secs_f64()
            );
        }
    }

    if criterion_specific_rois {
        super::criterion_rois::measure_criterion_specific_rois(
            input,
            &station_evidence,
            load_source,
            roi_cache,
            &mut station_source_images,
            &mut mtf,
            &mut noise,
            &mut unmeasurable,
            &mut timing,
        );
    }

    #[cfg(test)]
    write_tone_owner_fidelity_oracle(input, &rois, load_source)
        .expect("write test-only current-run tone owner Source_RAW fidelity oracle");

    #[cfg(test)]
    write_gradient_source_oracle(input, &rois, load_source)
        .expect("write test-only gradient source oracle");

    #[cfg(test)]
    super::focus_candidate_capture::capture(input, load_source)
        .expect("bounded focus candidate capture");

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
    let mut effective_result =
        effective_pixel_coverage_streaming(input.coverage, &quads, input.world_origin);
    if super::native_coverage::applicable(input) {
        effective_result = effective_result.and_then(|legacy| {
            super::native_coverage::measure(input, &legacy).map(|(result, audit)| {
                effective.effective_source_domain = Some(audit);
                result
            })
        });
    }
    #[cfg(test)]
    write_effective_coverage_diagnostics(input.coverage, &quads, input.world_origin);
    timing.effective_coverage_seconds = effective_started.elapsed().as_secs_f64();
    match effective_result {
        Ok(EffectivePixelCoverage {
            output_nontransparent,
            projected_union,
            ratio,
        }) => {
            effective.effective_output_pixels = Some(output_nontransparent);
            effective.effective_projected_pixels = Some(projected_union);
            effective.push(
                input.world_origin,
                ratio,
                "",
                ratio >= quality_gate::QUALITY_EFFECTIVE_PIXEL_RATIO_MIN,
            );
        }
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
                match quality_gate::measure_boundary_strokes_with_coverage(
                    input.output,
                    Some(input.coverage),
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
            input.acceptance_render_scale,
            criterion_specific_rois,
            &mut boundary,
            &mut unmeasurable,
        );
    }
    timing.boundary_seconds = boundary_started.elapsed().as_secs_f64();
    timing.boundary_sample_count = boundary.observed_count + boundary.unmeasurable;

    // Dense criteria keep the shared station windows. Independent MTF/noise
    // searches and streaming boundary cells already own their population
    // counts; never overwrite those with the eight dense ROI count.
    let roi_selection_mode = if timing.station_owner_evidence_roi_count > 0 {
        "station_evidence_shared_roi"
    } else {
        "raw_owner_shared_roi"
    };
    local_scale.set_candidate_pool(rois.len(), local_scale.observed_count, roi_selection_mode);
    local_ratio.set_candidate_pool(rois.len(), local_ratio.observed_count, roi_selection_mode);
    if !criterion_specific_rois {
        mtf.set_candidate_pool(rois.len(), mtf.observed_count, roi_selection_mode);
        noise.set_candidate_pool(rois.len(), noise.observed_count, roi_selection_mode);
    }
    gradient.set_candidate_pool(rois.len(), gradient.observed_count, roi_selection_mode);
    delta.set_candidate_pool(rois.len(), delta.observed_count, roi_selection_mode);
    if regions.is_some() {
        boundary.set_candidate_pool(
            boundary.observed_count + boundary.unmeasurable,
            boundary.observed_count,
            "owner_boundary_candidates",
        );
    }

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
    // Content-specific inapplicability (for example a low-contrast seam or a
    // ROI without a slanted edge) is already counted per criterion in
    // `content_not_applicable_count`.  Keep the report-level `unmeasurable`
    // list reserved for technical failures so a large, valid artwork does not
    // look broken merely because every criterion is not applicable to every
    // pixel region.  The criterion reason histograms retain the exact counts.
    report.unmeasurable = unmeasurable
        .into_iter()
        .filter(|record| record.category == super::super::report::UnmeasurableCategory::Technical)
        .collect();
    let cache_stats = roi_cache.stats();
    timing.roi_cache_hits = cache_stats.hits;
    timing.roi_cache_misses = cache_stats.misses;
    timing.roi_cache_evictions = cache_stats.evictions;
    timing.roi_cache_resident_bytes = cache_stats.resident_bytes;
    timing.roi_cache_peak_bytes = cache_stats.peak_bytes;
    report.timing = timing;
    #[cfg(test)]
    source_visual_capture::finish(&mut source_visual_capture, &report);
    report
}

/// Collapse the raw owner legend into the station identity that owns the
/// delivered geometry.  This is deliberately a selection-only plane: the
/// returned ROIs still carry the raw owner at their origin, so reference
/// sampling and provenance remain Source_RAW based.
fn station_evidence_owners(ownership: &[u16], sources: &[QualitySource]) -> Vec<u16> {
    let station_by_owner = sources
        .iter()
        .map(|source| {
            (
                source.owner,
                u16::try_from(source.geometry.station_id.saturating_add(1)).unwrap_or(u16::MAX),
            )
        })
        .collect::<BTreeMap<_, _>>();
    ownership
        .iter()
        .map(|owner| station_by_owner.get(owner).copied().unwrap_or_default())
        .collect()
}

fn select_station_level_rois(
    input: &QualityGateInput<'_>,
    _large_plane: bool,
    station_evidence: &[u16],
) -> Vec<QualityRoi> {
    let candidates = select_sparse_rois_with_evidence_side(
        input.coverage,
        input.ownership,
        station_evidence,
        input.world_origin,
        quality_gate::QUALITY_ROI_SIDE / 2,
    );
    let (width, height) = input.coverage.dimensions();
    let candidates = candidates
        .into_iter()
        .filter_map(|mut roi| {
            let index = (roi.y as usize)
                .checked_mul(width as usize)?
                .checked_add(roi.x as usize)?;
            let raw_owner = *input.ownership.get(index)?;
            let station_label = *station_evidence.get(index)?;
            if raw_owner == 0 || station_label == 0 {
                return None;
            }
            if roi.x >= width || roi.y >= height {
                return None;
            }
            let station_id = usize::from(station_label.saturating_sub(1));
            roi.owner = raw_owner;
            roi.region_label = quality_gate::station_level_roi_label(station_id)?;
            Some(roi)
        })
        .collect::<Vec<_>>();

    // The station plane is the evidence source, so a dense lattice does not
    // add independent information after the first few windows per station.
    // Keep a deterministic, evenly spread sample with at least one window
    // for every station.  This bounds RAW resampling in the real 9-source
    // acceptance while retaining the Quality_Gate minimum of eight ROIs.
    const STATION_ROI_MAX_COUNT: usize = quality_gate::QUALITY_MIN_MEASURABLE;
    if candidates.len() <= STATION_ROI_MAX_COUNT {
        return candidates;
    }
    let mut selected = BTreeSet::<usize>::new();
    for (index, roi) in candidates.iter().enumerate() {
        if selected
            .iter()
            .any(|selected_index| candidates[*selected_index].region_label == roi.region_label)
        {
            continue;
        }
        selected.insert(index);
    }
    if selected.len() > STATION_ROI_MAX_COUNT {
        return selected
            .into_iter()
            .take(STATION_ROI_MAX_COUNT)
            .map(|index| candidates[index].clone())
            .collect();
    }
    let remaining = STATION_ROI_MAX_COUNT - selected.len();
    for slot in 0..remaining {
        let index = slot.saturating_mul(candidates.len().saturating_sub(1))
            / remaining.saturating_sub(1).max(1);
        selected.insert(index);
    }
    for index in 0..candidates.len() {
        if selected.len() >= STATION_ROI_MAX_COUNT {
            break;
        }
        selected.insert(index);
    }
    selected
        .into_iter()
        .take(STATION_ROI_MAX_COUNT)
        .map(|index| candidates[index].clone())
        .collect()
}
