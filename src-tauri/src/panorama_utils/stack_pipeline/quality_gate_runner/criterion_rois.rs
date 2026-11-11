//! Independent, bounded searches for the content required by sparse metrics.
//!
//! Dense geometry/detail ROIs remain fixed. MTF and noise search their own
//! station-owned lattice on the delivered plane, then replay Source_RAW only
//! for eligible windows. A failed metric is never used to select another ROI.

use super::super::quality_gate::QualityRoi;
use super::super::report::{QualityGateTimingRecord, UnmeasurableCategory, UnmeasurableRecord};
use super::super::{degradation, quality_gate};
use super::coverage::sparse_rois_for_side;
use super::{
    Criterion, QualityGateInput, QualityRoiCache, SourceLoader, apply_run_tone_transform,
    classify_unmeasurable_reason, measure_station_local_scale, output_roi, ratio,
    record_unmeasurable, source_roi_for_station_quality, world,
};
use image::{GrayImage, Luma, Rgb32FImage};
use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;

// Candidate enumeration is capped by sparse_rois_for_side (256 per side).
// At most 24 reference windows per criterion enter the RAW sampler, keeping
// decoded-source reuse and the 64 MiB ROI cache effective on large canvases.
const MAX_SELECTED_PER_CRITERION: usize = 24;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SparseMetric {
    Mtf,
    Noise,
}

struct SelectedRoi {
    roi: QualityRoi,
    output: Rgb32FImage,
    metric: SparseMetric,
}

/// Round-robin station interiors before testing image content. In particular,
/// reaching the reference budget must not silently inspect only the top-left
/// station of a wide output. No metric result participates in this ordering.
fn candidates_for_side(
    input: &QualityGateInput<'_>,
    evidence: &[u16],
    side: u32,
) -> Vec<QualityRoi> {
    let mut stations = BTreeMap::<u16, VecDeque<QualityRoi>>::new();
    for mut roi in sparse_rois_for_side(
        input.coverage,
        input.ownership,
        evidence,
        input.world_origin,
        side,
    ) {
        let index = roi.y as usize * input.coverage.width() as usize + roi.x as usize;
        let Some(&label) = evidence.get(index).filter(|&&label| label != 0) else {
            continue;
        };
        let Some(region) = quality_gate::station_level_roi_label(usize::from(label - 1)) else {
            continue;
        };
        roi.region_label = region;
        stations.entry(label).or_default().push_back(roi);
    }
    let mut result = Vec::new();
    loop {
        let before = result.len();
        for candidates in stations.values_mut() {
            if let Some(roi) = candidates.pop_front() {
                result.push(roi);
            }
        }
        if before == result.len() {
            break;
        }
    }
    result
}

fn content_contract(image: &Rgb32FImage, metric: SparseMetric) -> Result<(), &'static str> {
    // Invalid pixels are a technical data failure. They must not disappear as
    // a "not slanted" or "not flat" selection rejection.
    if image
        .pixels()
        .any(|pixel| pixel.0.iter().any(|value| !value.is_finite()))
    {
        return Err(degradation::DIAGNOSTICS_ROI_INVALID);
    }
    match metric {
        SparseMetric::Mtf => quality_gate::detect_slanted_edge(image).map(|_| ()),
        SparseMetric::Noise => {
            let full = GrayImage::from_pixel(image.width(), image.height(), Luma([255]));
            quality_gate::flat_roi(image, &full).map(|_| ())
        }
    }
}

fn reject_or_fail(
    criterion: &mut Criterion,
    records: &mut Vec<UnmeasurableRecord>,
    roi: &QualityRoi,
    reason: &'static str,
) {
    if classify_unmeasurable_reason(reason) == UnmeasurableCategory::ContentNotApplicable {
        criterion.reject_content(reason, world(roi), 1);
    } else {
        record_unmeasurable(criterion, records, roi, reason);
    }
}

fn select_content_rois(
    input: &QualityGateInput<'_>,
    evidence: &[u16],
    metric: SparseMetric,
    criterion: &mut Criterion,
    records: &mut Vec<UnmeasurableRecord>,
) -> Vec<SelectedRoi> {
    let sides: &[u32] = match metric {
        // A 128px window cannot contain the required 128px contiguous edge
        // plus support. Search 256 first, then a larger 512px window.
        SparseMetric::Mtf => &[256, 512],
        SparseMetric::Noise => &[128],
    };
    let candidates = sides
        .iter()
        .flat_map(|&side| candidates_for_side(input, evidence, side))
        .collect::<Vec<_>>();
    let count = candidates.len();
    let mut selected = Vec::new();
    criterion.reference_roi_limit = MAX_SELECTED_PER_CRITERION;
    for (index, roi) in candidates.into_iter().enumerate() {
        if selected.len() == MAX_SELECTED_PER_CRITERION {
            criterion.unvisited_candidate_count = count - index;
            return selected;
        }
        criterion.candidate_count += 1;
        let Some(output) = output_roi(input.output, &roi) else {
            record_unmeasurable(
                criterion,
                records,
                &roi,
                degradation::DIAGNOSTICS_ROI_INVALID,
            );
            continue;
        };
        match content_contract(&output, metric) {
            Ok(()) => selected.push(SelectedRoi {
                roi,
                output,
                metric,
            }),
            Err(reason) => reject_or_fail(criterion, records, &roi, reason),
        }
    }
    selected
}

/// Search and measure sparse criteria independently of the dense ROI list.
/// References go through the existing bounded ROI cache and production
/// source loader; only eligible output candidates incur source resampling.
pub(super) fn measure_criterion_specific_rois(
    input: &QualityGateInput<'_>,
    evidence: &[u16],
    load_source: &mut SourceLoader<'_>,
    roi_cache: &mut QualityRoiCache,
    station_images: &mut BTreeMap<u16, Rgb32FImage>,
    mtf: &mut Criterion,
    noise: &mut Criterion,
    records: &mut Vec<UnmeasurableRecord>,
    timing: &mut QualityGateTimingRecord,
) {
    mtf.selection_mode = "station_slanted_edge_search_256_512".to_string();
    noise.selection_mode = "station_flat_patch_search_128".to_string();
    let started = Instant::now();
    let mut selected = select_content_rois(input, evidence, SparseMetric::Mtf, mtf, records);
    timing.slanted_edge_mtf_seconds += started.elapsed().as_secs_f64();
    let started = Instant::now();
    selected.extend(select_content_rois(
        input,
        evidence,
        SparseMetric::Noise,
        noise,
        records,
    ));
    timing.noise_seconds += started.elapsed().as_secs_f64();
    let sources = input
        .sources
        .iter()
        .map(|source| (source.owner, source))
        .collect::<BTreeMap<_, _>>();
    let mut by_owner = BTreeMap::<(usize, u16), Vec<SelectedRoi>>::new();
    for candidate in selected {
        let station = quality_gate::station_level_roi_station_id(candidate.roi.region_label)
            .unwrap_or(usize::MAX);
        by_owner
            .entry((station, candidate.roi.owner))
            .or_default()
            .push(candidate);
    }
    for ((_station, owner), candidates) in by_owner {
        let Some(source) = sources.get(&owner).copied() else {
            for candidate in candidates {
                let criterion = if candidate.metric == SparseMetric::Mtf {
                    &mut *mtf
                } else {
                    &mut *noise
                };
                record_unmeasurable(
                    criterion,
                    records,
                    &candidate.roi,
                    degradation::OWNER_SOURCE_UNDECODABLE,
                );
            }
            continue;
        };
        // One primary load per owner, only on a cache miss. Other station
        // members are loaded by the existing station reference cache.
        let mut primary = None;
        for candidate in candidates {
            let started = Instant::now();
            let criterion = if candidate.metric == SparseMetric::Mtf {
                &mut *mtf
            } else {
                &mut *noise
            };
            let roi = &candidate.roi;
            timing.roi_reference_count += 1;
            let cached = roi_cache.take(source, roi);
            let reference = if let Some(reference) = cached {
                Ok(reference)
            } else {
                if primary.is_none() {
                    let decode_started = Instant::now();
                    primary = Some(
                        load_source(source).map_err(|_| degradation::OWNER_SOURCE_UNDECODABLE),
                    );
                    timing.decode_seconds += decode_started.elapsed().as_secs_f64();
                    if primary.as_ref().is_some_and(Result::is_ok) {
                        timing.owner_source_decode_count += 1;
                    }
                }
                primary
                    .as_ref()
                    .unwrap()
                    .as_ref()
                    .map_err(|&reason| reason)
                    .and_then(|image| {
                        source_roi_for_station_quality(
                            station_images,
                            owner,
                            Some(image),
                            input.ownership,
                            input.coverage.width(),
                            roi,
                            source.geometry.station_id,
                            &sources,
                            input.residual,
                            load_source,
                        )
                    })
                    .map(|mut reference| {
                        if source.native_registration.is_none() {
                            apply_run_tone_transform(&mut reference, source.geometry.station_id);
                        }
                        reference
                    })
            };
            timing.reference_resampling_seconds += started.elapsed().as_secs_f64();
            let reference = match reference {
                Ok(reference) => reference,
                Err(reason) => {
                    record_unmeasurable(criterion, records, roi, reason);
                    continue;
                }
            };
            let metric_started = Instant::now();
            let result = content_contract(&reference, candidate.metric).and_then(|()| {
                match candidate.metric {
                    SparseMetric::Mtf => {
                        let local = measure_station_local_scale(
                            roi,
                            input.ownership,
                            input.coverage.width(),
                            source.geometry.station_id,
                            &sources,
                            input.residual,
                            input.acceptance_render_scale,
                        )?
                        .median;
                        let output = quality_gate::slanted_edge_mtf50(&candidate.output, local)?;
                        let reference = quality_gate::slanted_edge_mtf50(&reference, local)?;
                        ratio(
                            output.normalized,
                            reference.normalized,
                            quality_gate::QUALITY_MTF50_RATIO_MIN,
                            f64::INFINITY,
                        )
                        .ok_or("mtf_reference_zero")
                    }
                    SparseMetric::Noise => {
                        quality_gate::noise_sigma_ratio(&candidate.output, &reference).map(
                            |value| {
                                (
                                    value,
                                    (quality_gate::QUALITY_NOISE_RATIO_MIN
                                        ..=quality_gate::QUALITY_NOISE_RATIO_MAX)
                                        .contains(&value)
                                        && input.final_sharpen_amount <= 0.0,
                                )
                            },
                        )
                    }
                }
            });
            match result {
                Ok((value, pass)) => {
                    criterion.eligible_count += 1;
                    criterion.push(world(roi), value, &source.path, pass);
                }
                Err(reason) => reject_or_fail(criterion, records, roi, reason),
            }
            match candidate.metric {
                SparseMetric::Mtf => {
                    timing.slanted_edge_mtf_seconds += metric_started.elapsed().as_secs_f64()
                }
                SparseMetric::Noise => {
                    timing.noise_seconds += metric_started.elapsed().as_secs_f64()
                }
            }
            roi_cache.insert(source, roi, reference);
            timing.roi_sampling_seconds += started.elapsed().as_secs_f64();
        }
    }
}
