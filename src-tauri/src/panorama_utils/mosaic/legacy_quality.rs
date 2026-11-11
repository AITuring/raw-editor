//! Observation-only native evidence for the source-scan compositor.
//!
//! Geometry and owner IDs come from the actual sampler and tiled writes.
//! Candidate sharpness is measured on a common 64px canvas lattice using the
//! existing five-probe / 32px native-window metric. It never changes a seam.
use super::{LayerSampler, NativeResidualDisplacement, StreamingMosaicStore};
use crate::panorama_utils::stack_pipeline::{
    focus_fuser,
    quality_gate::SourceGeometry,
    quality_gate_runner::{self, NativeReferenceMap, QualitySource},
};
use crate::panorama_utils::stitching::{FocusSourceRefinement, FocusSourceSampling, Projection};
use image::{GrayImage, Rgb};
use nalgebra::Matrix3;
use rayon::prelude::*;
use std::collections::BTreeMap;

const CELL: u32 = 64;

#[derive(Debug, Clone)]
pub(crate) struct LegacyQualityObservation {
    pub ownership: Vec<u16>,
    pub confidence: Vec<f32>,
    pub textured: Vec<u8>,
    pub shortfall: Vec<u8>,
    pub disagreement: Vec<u8>,
    pub unresolved: u64,
    pub origin: (f64, f64),
    pub sources: Vec<QualitySource>,
    pub confidence_scores:
        crate::panorama_utils::stack_pipeline::report::ConfidenceScoreDistribution,
    pub report: crate::panorama_utils::stack_pipeline::report::LegacyNativeEvidenceReport,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    station: u16,
    owner: u16,
    score: f32,
    colour: Rgb<f32>,
}

/// A cell the focus-scan detail repair rewrote, in canvas pixels.
pub(super) struct DetailRepairRegion {
    /// `(x0, y0, x1, y1)`, half-open.
    pub(super) canvas_rect: (u32, u32, u32, u32),
    pub(super) donor_source_id: usize,
    pub(super) donor_path: String,
    pub(super) outcome: &'static str,
}

#[derive(Default)]
pub(super) struct LegacyQualityCapture {
    cells: BTreeMap<(u32, u32), Vec<Candidate>>,
    sources: Vec<QualitySource>,
    repairs: Vec<DetailRepairRegion>,
}

impl LegacyQualityCapture {
    /// Record the regions the detail repair rewrote after ownership was final.
    pub(super) fn record_detail_repairs(&mut self, regions: Vec<DetailRepairRegion>) {
        self.repairs = regions;
    }

    pub(super) fn observe(
        &mut self,
        owner: u8,
        capture_station_id: u8,
        sampler: &LayerSampler<'_>,
        write_tone: &super::Field<3>,
        right: u32,
        bottom: u32,
    ) {
        let plan = focus_fuser::CellSamplingPlan::for_cell(f64::from(CELL));
        let columns: Vec<_> = (sampler.left / CELL..=right / CELL).collect();
        let observations: Vec<_> = (sampler.top / CELL..=bottom / CELL)
            .into_par_iter()
            .flat_map_iter(|row| {
                let plan = &plan;
                columns.iter().filter_map(move |&column| {
                    let x = f64::from(column * CELL) - f64::from(sampler.left);
                    let y = f64::from(row * CELL) - f64::from(sampler.top);
                    let colour =
                        sampler.sample(x + f64::from(CELL) * 0.5, y + f64::from(CELL) * 0.5)?;
                    let score = plan.measure(x, y, |sx, sy| sampler.sample(sx, sy)) as f32;
                    Some((
                        (column, row),
                        Candidate {
                            station: u16::from(capture_station_id),
                            owner: u16::from(owner),
                            score,
                            colour,
                        },
                    ))
                })
            })
            .collect();
        for (cell, candidate) in observations {
            self.cells.entry(cell).or_default().push(candidate);
        }
        // Store the direct output->RAW map, NOT a twice-resampled station layer.
        let mut into_canvas = Matrix3::identity();
        into_canvas[(0, 2)] = sampler.offset.0;
        into_canvas[(1, 2)] = sampler.offset.1;
        let mut to_world = Matrix3::identity();
        to_world[(0, 2)] = -sampler.offset.0;
        to_world[(1, 2)] = -sampler.offset.1;
        let transform = sampler
            .inverse
            .try_inverse()
            .expect("validated source inverse");
        self.sources.push(QualitySource {
            owner: u16::from(owner),
            path: sampler.info.filename.clone(),
            dimensions: sampler.info.dimensions(),
            geometry: SourceGeometry {
                station_id: usize::from(capture_station_id),
                member_to_anchor: Matrix3::identity(),
                tile_to_world: transform,
            },
            native_warp: None,
            native_registration: Some(NativeReferenceMap {
                mosaic_tone: Some(super::MosaicReferenceTone {
                    field: write_tone.clone(),
                    left: sampler.left,
                    top: sampler.top,
                    scale: sampler.scale,
                    offset: sampler.offset,
                    post_gain: None,
                }),
                world_to_station: into_canvas,
                station_to_world: to_world,
                station_origin: (0.0, 0.0),
                source_prefilter_divisor: sampler.source_divisor,
                refinement: FocusSourceRefinement {
                    displacement: Some(NativeResidualDisplacement {
                        field: sampler.residual.clone(),
                        left: sampler.left,
                        top: sampler.top,
                        scale: sampler.scale,
                    }),
                    source_sampling: Some(FocusSourceSampling::for_mosaic(
                        sampler.info,
                        &(into_canvas * transform),
                        Projection::Planar,
                    )),
                    ..Default::default()
                },
            }),
        });
    }

    pub(super) fn finish(
        mut self,
        store: &StreamingMosaicStore,
        crop: (u32, u32, u32, u32),
        coverage: &GrayImage,
        offset: (f64, f64),
        post_gain: Option<super::MosaicGainReference>,
        load_source: &mut quality_gate_runner::SourceLoader<'_>,
    ) -> Result<LegacyQualityObservation, String> {
        if coverage.dimensions() != (crop.2, crop.3) {
            return Err("legacy evidence crop differs from delivered coverage".into());
        }
        for source in &mut self.sources {
            if let Some(tone) = source
                .native_registration
                .as_mut()
                .and_then(|native| native.mosaic_tone.as_mut())
            {
                tone.post_gain = post_gain.clone();
            }
        }
        let len = coverage.as_raw().len();
        let unknown = quality_gate_runner::OWNER_EVIDENCE_UNKNOWN;
        use crate::panorama_utils::stack_pipeline::report::{
            LegacyNativeEvidenceReport, LegacyNativeSourceRecord,
        };
        let origin = (f64::from(crop.0) - offset.0, f64::from(crop.1) - offset.1);
        let native_report = LegacyNativeEvidenceReport {
            method: "streaming_native_owner_direct_sampler_five_probe_v1".into(),
            accepted_source_pairs: Vec::new(),
            focus_population: "actual_capture_station_members_only".into(),
            capture_station_count: self
                .sources
                .iter()
                .map(|source| source.geometry.station_id)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            adaptive_pixel_count: 0,
            adaptive_cell_count: 0,
            adaptive_source_load_count: 0,
            cell_size_px: CELL,
            sample_window_px: focus_fuser::CellSamplingPlan::for_cell(f64::from(CELL)).window_px(),
            sampled_cell_count: self.cells.len(),
            candidate_measurement_count: self.cells.values().map(Vec::len).sum(),
            delivered_pixels: len,
            unresolved_pixel_count: 0,
            world_origin: origin,
            detail_repairs: Vec::new(),
            sources: self
                .sources
                .iter()
                .map(|source| {
                    let native = source
                        .native_registration
                        .as_ref()
                        .expect("captured native source");
                    let displacement = native
                        .refinement
                        .displacement
                        .as_ref()
                        .expect("captured residual field");
                    let matrix = source.geometry.tile_to_world;
                    LegacyNativeSourceRecord {
                        owner: source.owner,
                        source_id: usize::from(source.owner) - 1,
                        capture_station_id: source.geometry.station_id,
                        source_tone_replayed: native.mosaic_tone.is_some(),
                        owner_gain_replayed: native
                            .mosaic_tone
                            .as_ref()
                            .is_some_and(|tone| tone.post_gain.is_some()),
                        path: source.path.clone(),
                        dimensions: source.dimensions,
                        source_to_world: (0..3)
                            .flat_map(|row| (0..3).map(move |column| matrix[(row, column)]))
                            .collect(),
                        residual_origin: (displacement.left, displacement.top),
                        residual_scale: displacement.scale,
                        residual_nodes: (displacement.field.width, displacement.field.height),
                        residual_max_offset_px: displacement.max_offset_px(),
                        source_prefilter_divisor: native.source_prefilter_divisor,
                    }
                })
                .collect(),
        };
        let mut result = LegacyQualityObservation {
            ownership: vec![0; len],
            confidence: vec![f32::NAN; len],
            textured: vec![0; len],
            shortfall: vec![unknown; len],
            disagreement: vec![unknown; len],
            unresolved: 0,
            origin: (f64::from(crop.0) - offset.0, f64::from(crop.1) - offset.1),
            sources: self.sources,
            confidence_scores:
                crate::panorama_utils::stack_pipeline::report::ConfidenceScoreDistribution {
                    winner_histogram: vec![0; 21],
                    runner_up_histogram: vec![0; 21],
                    ..Default::default()
                },
            report: native_report,
        };
        result.report.detail_repairs = self
            .repairs
            .iter()
            .filter_map(|region| {
                let (x0, y0, x1, y1) = region.canvas_rect;
                let left = x0.max(crop.0);
                let top = y0.max(crop.1);
                let right = x1.min(crop.0 + crop.2);
                let bottom = y1.min(crop.1 + crop.3);
                (left < right && top < bottom).then(|| {
                    crate::panorama_utils::stack_pipeline::report::LegacyDetailRepairRecord {
                        delivered_rect: (left - crop.0, top - crop.1, right - left, bottom - top),
                        donor_source_id: region.donor_source_id,
                        donor_path: region.donor_path.clone(),
                        outcome: region.outcome.into(),
                    }
                })
            })
            .collect();
        let mut unknown_cells = BTreeMap::new();
        let mut by_cell = BTreeMap::new();
        for (cell, candidates) in self.cells {
            for own in &candidates {
                by_cell
                    .entry(cell)
                    .or_insert_with(Vec::new)
                    .push((own.owner, evidence(&candidates, own.owner)));
            }
        }
        // Read exact native owners, not the 2048px diagnostic PNG. Missing or
        // corrupt owner files are errors, never transparent/fabricated evidence.
        for row in crop.1 / store.tile_size..=(crop.1 + crop.3 - 1) / store.tile_size {
            for column in crop.0 / store.tile_size..=(crop.0 + crop.2 - 1) / store.tile_size {
                let (left, top, width, height) = store.tile_extent(column, row);
                let bytes = match std::fs::read(store.tile_path(column, row, "owner")) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        let covered =
                            (top.max(crop.1)..(top + height).min(crop.1 + crop.3)).any(|y| {
                                (left.max(crop.0)..(left + width).min(crop.0 + crop.2))
                                    .any(|x| coverage.get_pixel(x - crop.0, y - crop.1)[0] != 0)
                            });
                        if covered {
                            return Err(format!("legacy native owner read: {error}"));
                        }
                        continue;
                    }
                };
                if bytes.len() != width as usize * height as usize {
                    return Err("legacy native owner tile length mismatch".into());
                }
                for y in top.max(crop.1)..(top + height).min(crop.1 + crop.3) {
                    let mut previous_column = u32::MAX;
                    let mut cell_evidence = [None; 256];
                    for x in left.max(crop.0)..(left + width).min(crop.0 + crop.2) {
                        if x / CELL != previous_column {
                            previous_column = x / CELL;
                            cell_evidence.fill(None);
                            if let Some(candidates) = by_cell.get(&(x / CELL, y / CELL)) {
                                for &(owner, evidence) in candidates {
                                    cell_evidence[owner as usize] = Some(evidence);
                                }
                            }
                        }
                        let index = (y - crop.1) as usize * crop.2 as usize + (x - crop.0) as usize;
                        if coverage.as_raw()[index] == 0 {
                            continue;
                        }
                        let owner = u16::from(
                            bytes[(y - top) as usize * width as usize + (x - left) as usize],
                        );
                        result.ownership[index] = owner;
                        if let Some(&(confidence, textured, shortfall, disagreement, scores)) =
                            cell_evidence[owner as usize].as_ref()
                        {
                            result.confidence[index] = confidence;
                            result.textured[index] = textured;
                            result.shortfall[index] = shortfall;
                            result.disagreement[index] = disagreement;
                            quality_gate_runner::accumulate_confidence_score(
                                &mut result.confidence_scores,
                                255,
                                confidence,
                                textured,
                                scores,
                            );
                        } else {
                            result.unresolved += 1;
                            unknown_cells.entry((x / CELL, y / CELL, owner)).or_insert(
                                nalgebra::Point2::new(
                                    f64::from(x) - offset.0,
                                    f64::from(y) - offset.1,
                                ),
                            );
                        }
                    }
                }
            }
        }
        result.report.adaptive_cell_count = unknown_cells.len();
        result.report.adaptive_pixel_count = result.unresolved;
        if !unknown_cells.is_empty() {
            // The coarse centre can lie outside a source although that source
            // really owns a small part of the cell. Re-measure at an actual
            // owned pixel; every same-station candidate uses the SAME five
            // probes and 32px support. Never infer a missing owner from RGB.
            let station_by_owner: BTreeMap<_, _> = result
                .sources
                .iter()
                .map(|source| (source.owner, source.geometry.station_id))
                .collect();
            let mut adaptive = BTreeMap::<_, Vec<Candidate>>::new();
            let plan = focus_fuser::CellSamplingPlan::for_cell(f64::from(CELL));
            let residual =
                crate::panorama_utils::stack_pipeline::residual_warp::ResidualWarp::default();
            for source in &result.sources {
                let requests: Vec<_> = unknown_cells
                    .iter()
                    .filter(|(key, _)| {
                        station_by_owner.get(&key.2) == Some(&source.geometry.station_id)
                    })
                    .collect();
                if requests.is_empty() {
                    continue;
                }
                let mut geometry_source = source.clone();
                geometry_source
                    .native_registration
                    .as_mut()
                    .expect("native map")
                    .mosaic_tone = None;
                let image = quality_gate_runner::prefilter_source_reference(
                    load_source(&geometry_source)?,
                    &geometry_source,
                )?;
                result.report.adaptive_source_load_count += 1;
                for (&key, &world) in requests {
                    let sample = |x, y| {
                        quality_gate_runner::sample_quality_source_at_world(
                            &image,
                            nalgebra::Point2::new(x, y),
                            &geometry_source,
                            &residual,
                        )
                        .ok()
                    };
                    let Some(colour) = sample(world.x, world.y) else {
                        continue;
                    };
                    let score = plan.measure(
                        world.x - f64::from(CELL) * 0.5,
                        world.y - f64::from(CELL) * 0.5,
                        sample,
                    ) as f32;
                    adaptive.entry(key).or_default().push(Candidate {
                        station: source.geometry.station_id as u16,
                        owner: source.owner,
                        score,
                        colour,
                    });
                }
            }
            let adaptive: BTreeMap<_, _> = adaptive
                .into_iter()
                .filter_map(|(key, candidates)| {
                    candidates
                        .iter()
                        .any(|c| c.owner == key.2)
                        .then(|| (key, evidence(&candidates, key.2)))
                })
                .collect();
            for index in 0..len {
                if coverage.as_raw()[index] == 0 || result.shortfall[index] != unknown {
                    continue;
                }
                let x = crop.0 + index as u32 % crop.2;
                let y = crop.1 + index as u32 / crop.2;
                if let Some(&(confidence, textured, shortfall, disagreement, scores)) =
                    adaptive.get(&(x / CELL, y / CELL, result.ownership[index]))
                {
                    result.confidence[index] = confidence;
                    result.textured[index] = textured;
                    result.shortfall[index] = shortfall;
                    result.disagreement[index] = disagreement;
                    quality_gate_runner::accumulate_confidence_score(
                        &mut result.confidence_scores,
                        255,
                        confidence,
                        textured,
                        scores,
                    );
                    result.unresolved -= 1;
                }
            }
        }
        result.report.unresolved_pixel_count = result.unresolved;
        Ok(result)
    }
}

fn evidence(
    candidates: &[Candidate],
    owner: u16,
) -> (f32, u8, u8, u8, focus_fuser::SharpnessCellEvidence) {
    let own = candidates
        .iter()
        .find(|c| c.owner == owner)
        .expect("observed owner");
    let candidates: Vec<_> = candidates
        .iter()
        .filter(|candidate| candidate.station == own.station)
        .collect();
    let mut scores: Vec<_> = candidates.iter().map(|c| c.score).collect();
    scores.sort_by(|a, b| b.total_cmp(a));
    let best = scores[0];
    let second = scores.get(1).copied().unwrap_or(0.0);
    let disagreement = candidates
        .iter()
        .map(|c| {
            (0..3)
                .map(|channel| (own.colour[channel] - c.colour[channel]).abs())
                .sum::<f32>()
                / 3.0
        })
        .fold(0.0, f32::max);
    let scores = focus_fuser::SharpnessCellEvidence {
        winner_score: best,
        runner_up_score: second,
        owner_score: own.score,
        candidate_count: candidates.len() as u32,
        disagreement,
    };
    let shortfall = scores.owner_shortfall();
    (
        focus_fuser::cell_confidence(f64::from(best), f64::from(second), candidates.len()),
        if f64::from(best) >= focus_fuser::TEXTURED_PIXEL_SHARPNESS_FLOOR {
            255
        } else {
            0
        },
        quality_gate_runner::quantize_hundredths(shortfall),
        quality_gate_runner::quantize_hundredths(disagreement),
        scores,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn single_candidate_is_not_perfect_confidence() {
        let c = Candidate {
            station: 1,
            owner: 7,
            score: 0.8,
            colour: Rgb([0.5; 3]),
        };
        let (confidence, textured, shortfall, _, _) = evidence(&[c], 7);
        assert_eq!(confidence, 0.0);
        assert_eq!(textured, 255);
        assert_eq!(shortfall, 0);
    }
    #[test]
    fn reports_actual_losing_owner_without_reselecting_it() {
        let candidates = [
            Candidate {
                station: 1,
                owner: 7,
                score: 0.8,
                colour: Rgb([0.5; 3]),
            },
            Candidate {
                station: 1,
                owner: 9,
                score: 0.4,
                colour: Rgb([0.6; 3]),
            },
        ];
        let (confidence, _, shortfall, disagreement, _) = evidence(&candidates, 9);
        assert_eq!(confidence, 0.5);
        assert_eq!(shortfall, 50);
        assert!(disagreement >= 10);
    }
    #[test]
    fn other_camera_positions_are_not_focus_runner_ups() {
        let candidates = [
            Candidate {
                station: 1,
                owner: 7,
                score: 0.4,
                colour: Rgb([0.5; 3]),
            },
            Candidate {
                station: 2,
                owner: 9,
                score: 0.9,
                colour: Rgb([0.8; 3]),
            },
        ];
        let (confidence, textured, shortfall, disagreement, scores) = evidence(&candidates, 7);
        assert_eq!(confidence, 0.0);
        assert_eq!(textured, 255);
        assert_eq!(shortfall, 0);
        assert_eq!(disagreement, 0);
        assert_eq!(scores.candidate_count, 1);
        assert_eq!(scores.winner_score, 0.4);
        assert_eq!(scores.runner_up_score, 0.0);
    }
}

#[cfg(test)]
mod native_tests {
    use super::*;
    use crate::panorama_utils::{shared::ImageInfo, stack_pipeline::residual_warp::ResidualWarp};
    use image::{Luma, Rgb32FImage};
    fn info(image: &Rgb32FImage) -> ImageInfo {
        ImageInfo {
            id: 6,
            filename: "source-native.png".into(),
            width: image.width(),
            height: image.height(),
            alignment_image: image::DynamicImage::ImageRgb32F(image.clone()).to_luma8(),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm: None,
            source_footprint: None,
            overview_reference: false,
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: Vec::new(),
            vertical_edge_columns: Vec::new(),
        }
    }
    #[test]
    fn direct_reference_replays_offset_local_field_and_actual_prefilter() {
        let original = Rgb32FImage::from_fn(180, 160, |x, y| {
            Rgb([
                0.2 + (x % 13) as f32 * 0.02,
                0.3 + (y % 17) as f32 * 0.01,
                0.5,
            ])
        });
        let info = info(&original);
        for divisor in [1.0, 2.0] {
            let source = if divisor == 1.0 {
                original.clone()
            } else {
                crate::panorama_utils::stitching::downsample_rgb_half(&original)
            };
            let transform = Matrix3::new(1.05, 0.0, 3.0, 0.0, 1.05, 8.0, 0.0, 0.0, 1.0);
            let mut residual = super::super::Field::new(100, 100, 16.0);
            residual.values.fill([0.3, -0.6]);
            let sampler = LayerSampler {
                info: &info,
                source: &source,
                source_divisor: divisor,
                inverse: transform.try_inverse().unwrap(),
                native_mapping: None,
                projection: Projection::Planar,
                offset: (37.0, -19.0),
                left: 40,
                top: 0,
                scale: 2.1,
                residual,
            };
            let mut capture = LegacyQualityCapture::default();
            capture.observe(
                7,
                7,
                &sampler,
                &super::super::Field::new(100, 100, 112.0),
                180,
                140,
            );
            let native = &capture.sources[0];
            let reference =
                quality_gate_runner::prefilter_source_reference(original.clone(), native).unwrap();
            for (x, y) in [(50.0, 70.0), (70.5, 80.25), (100.0, 90.0)] {
                let expected = sampler.sample(x, y).unwrap();
                let world = nalgebra::Point2::new(
                    f64::from(sampler.left) + x - sampler.offset.0,
                    f64::from(sampler.top) + y - sampler.offset.1,
                );
                let actual = quality_gate_runner::sample_quality_source_at_world(
                    &reference,
                    world,
                    native,
                    &ResidualWarp::default(),
                )
                .unwrap();
                for c in 0..3 {
                    assert!(
                        (actual[c] - expected[c]).abs() < 1e-6,
                        "divisor {divisor}: {actual:?} != {expected:?}"
                    );
                }
            }
        }
    }
    #[test]
    fn native_crop_preserves_owner_ids_holes_and_unknown_measurements() {
        let store = StreamingMosaicStore::new(1030, 70).unwrap();
        for column in 0..store.tile_columns {
            let (left, _, width, height) = store.tile_extent(column, 0);
            let owners = GrayImage::from_fn(width, height, |x, _| {
                Luma([if left + x < 1024 { 7 } else { 9 }])
            });
            store.save_owner_tile(column, 0, &owners).unwrap();
        }
        let crop = (1000, 10, 30, 20);
        let mut coverage = GrayImage::from_pixel(30, 20, Luma([255]));
        coverage.put_pixel(0, 0, Luma([0]));
        let mut capture = LegacyQualityCapture::default();
        let c = Candidate {
            station: 1,
            owner: 7,
            score: 0.5,
            colour: Rgb([0.4; 3]),
        };
        capture.cells.insert((15, 0), vec![c]);
        let result = capture
            .finish(
                &store,
                crop,
                &coverage,
                (37.0, -19.0),
                None,
                &mut |_source| Err("unexpected native source load".into()),
            )
            .unwrap();
        assert_eq!(result.origin, (963.0, 29.0));
        assert_eq!(result.ownership[0], 0);
        assert_eq!(result.ownership[23], 7);
        assert_eq!(result.ownership[24], 9);
        assert_eq!(result.confidence[23], 0.0);
        assert!(result.confidence[24].is_nan());
        assert_eq!(
            result.shortfall[24],
            quality_gate_runner::OWNER_EVIDENCE_UNKNOWN
        );
        assert_eq!(result.unresolved, 6 * 20);
        assert_eq!(result.report.unresolved_pixel_count, 120);
    }
    #[test]
    fn missing_native_owner_backing_fails_closed() {
        let store = StreamingMosaicStore::new(8, 8).unwrap();
        assert!(
            LegacyQualityCapture::default()
                .finish(
                    &store,
                    (0, 0, 8, 8),
                    &GrayImage::from_pixel(8, 8, Luma([255])),
                    (0.0, 0.0),
                    None,
                    &mut |_source| Err("unexpected native source load".into()),
                )
                .is_err()
        );
    }
    #[test]
    fn partial_cell_remeasures_real_owned_support_without_extrapolated_confidence() {
        let image = Rgb32FImage::from_pixel(20, 64, Rgb([0.4; 3]));
        let info = info(&image);
        let sampler = LayerSampler {
            info: &info,
            source: &image,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            native_mapping: None,
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: super::super::Field::new(32, 64, 16.0),
        };
        let mut capture = LegacyQualityCapture::default();
        capture.observe(
            7,
            3,
            &sampler,
            &super::super::Field::new(32, 64, 112.0),
            19,
            63,
        );
        assert!(
            capture.cells.is_empty(),
            "coarse centre is outside this real narrow footprint"
        );
        let store = StreamingMosaicStore::new(32, 64).unwrap();
        let owners = GrayImage::from_pixel(32, 64, Luma([7]));
        store.save_owner_tile(0, 0, &owners).unwrap();
        let coverage = GrayImage::from_fn(32, 64, |x, y| {
            Luma([if (2..18).contains(&x) && (18..46).contains(&y) {
                255
            } else {
                0
            }])
        });
        let result = capture
            .finish(
                &store,
                (0, 0, 32, 64),
                &coverage,
                (0.0, 0.0),
                None,
                &mut |_source| Ok(image.clone()),
            )
            .unwrap();
        assert_eq!(result.unresolved, 0);
        assert_eq!(result.report.adaptive_cell_count, 1);
        assert_eq!(result.report.adaptive_source_load_count, 1);
        assert_eq!(result.report.adaptive_pixel_count, 16 * 28);
        for (index, &covered) in coverage.as_raw().iter().enumerate() {
            if covered != 0 {
                assert_eq!(result.ownership[index], 7);
                assert_eq!(result.confidence[index], 0.0);
                assert_eq!(result.shortfall[index], 0);
            }
        }
    }
}
