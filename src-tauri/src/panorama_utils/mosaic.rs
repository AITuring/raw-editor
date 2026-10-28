//! Focus mosaics: refine residual geometry, choose sharp source detail, match
//! only broad colour. The output is never an average of displaced fine detail.
use super::stack_pipeline::intra_station;
#[cfg(test)]
use super::stack_pipeline::report::IntraStationFrameStatus;
#[cfg(test)]
use super::stitching::downsample_rgb_half;
use super::stitching::{Projection, get_high_quality_interpolated_pixel};
use crate::panorama_utils::shared::ImageInfo;
#[cfg(test)]
use image::Luma;
use image::{GrayImage, Rgb, Rgb32FImage};
use nalgebra::Matrix3;
#[cfg(test)]
use nalgebra::{Point2, Point3};
#[cfg(test)]
use rayon::prelude::*;
#[cfg(test)]
use std::collections::HashMap;

#[cfg(test)]
#[path = "mosaic_diagnostics.rs"]
mod diagnostics;

mod sampling;

pub(crate) use sampling::Field;
use sampling::LayerSampler;
#[cfg(test)]
use sampling::is_lower_left_black_capture_border;

mod streaming_store;

#[cfg(test)]
use streaming_store::sample_decision_alpha;
use streaming_store::{
    StreamingMosaicStore, copy_streaming_owned_tile_region, copy_streaming_tile_region,
    fill_streaming_uncovered_tile_region, materialize_streaming_focus_output,
};

mod streaming_harmonization;

#[cfg(test)]
use streaming_harmonization::streaming_group_tone_relations_from_analysis;
use streaming_harmonization::{
    StreamingLowFrequencyConsensus, apply_streaming_seam_harmonization,
    streaming_seam_harmonization,
};

mod streaming_illumination;

use streaming_illumination::apply_streaming_source_owner_low_frequency_correction;
pub(super) use streaming_illumination::smooth_streaming_low_frequency_illumination;

mod layer_refine;

use layer_refine::{
    refine_layer, refine_layer_from_analysis, streaming_candidate_analysis,
    streaming_refinement_pair,
};

mod layer_tone;

use layer_tone::{streaming_tone_field_from_analysis, tone_field};

mod source_ownership;

use source_ownership::{ownership, ownership_with_long_side, streaming_ownership_from_analysis};
#[cfg(test)]
use source_ownership::{
    ownership_disagreement, ownership_grid, streaming_ownership_with_long_side,
    streaming_transition_structure,
};

mod focus_measure;

pub(super) use focus_measure::{
    ACUTANCE_PATCH_POINTS, acutance_with_step, cell_focus_with_step, cell_low_pass_differences,
    cell_probe_offsets,
};
use focus_measure::{cell_focus, streaming_cell_focus};
pub(crate) use focus_measure::{gray_analysis_sharpness, station_member_median_sharpness};

mod native_control_points;

use native_control_points::{SimilarityStats, collect_native_observations};
#[cfg(test)]
use native_control_points::{contrasted_native_control_location, measure_native_control_point};

mod native_refine;

pub(crate) use native_refine::NativeRefineMode;
#[cfg(test)]
pub(crate) use native_refine::native_control_point_spacing_px;
use native_refine::{
    NativeRefineBase, NativeRefinePass, intra_station_frame_record, refine_native_layer,
    refine_native_layer_streaming,
};

mod station_registration;

pub(crate) use station_registration::{
    NativeResidualDisplacement, StationFrameRegistration,
    register_station_frame_native_with_mapping,
};

mod in_memory_mosaic;

pub(crate) use in_memory_mosaic::detail_preserving_mosaic;
#[cfg(test)]
use in_memory_mosaic::mask_covered_bounds;

mod streaming_mosaic;

use streaming_mosaic::detail_preserving_mosaic_streaming;

// Actual configured intra-station analysis long side. Keep both compositor
// paths on the same source of truth so Stack_Report records what production
// used rather than the 2048 requirement ceiling (需求 2.2).
const ANALYSIS_LONG_SIDE: u32 = intra_station::INTRA_STATION_ANALYSIS_LONG_SIDE;
// Bound the grid per source layer, not by the full panorama width. At a
// roughly 9504px layer footprint this yields approximately 19px cells, fine
// enough for seams to route around individual strokes.
// Also the Focus_Fuser's ownership cell count (需求 3.1): the station plane's
// long side is divided into at least this many cells.
pub(super) const SELECTION_LONG_SIDE: u32 = 512;
const LONG_SEQUENCE_SELECTION_LONG_SIDE: u32 = 768;
// Streaming still makes one continuous decision for the whole source layer.
// Keeping the analysis bounded preserves memory while preventing independent
// 1024px tiles from choosing incompatible owners along their shared border.
const STREAMING_ANALYSIS_LONG_SIDE: u32 = intra_station::INTRA_STATION_ANALYSIS_LONG_SIDE;
const STREAMING_SELECTION_LONG_SIDE: u32 = 768;
// Ownership is already regularised by the graph cut. A wide blur here mixes
// focused and defocused focal planes across tens of native pixels on a 45MP
// source. Keep only a sub-cell antialiasing transition.
const STREAMING_OWNERSHIP_FEATHER: f32 = 0.65;
const STREAMING_HARMONIZATION_LONG_SIDE: u32 = 2400;
// Keep this narrow enough to cancel the actual source discontinuity at an
// ownership boundary. A broad gain blur follows nearby dark strokes and draws
// a visible halo even though the gain itself is constant per capture group.
const STREAMING_GROUP_GAIN_FEATHER: f32 = 3.0;
// Allow a bounded correction for a real source exposure step while rejecting
// the extreme ratios produced by a displaced brush stroke.  All relation
// paths (seam probes, direct overlap measurements, and the robust group solve)
// share this symmetric cap.
const STREAMING_GROUP_GAIN_MAX_LOG: f64 = 0.70;
// Camera-position lighting varies smoothly across a frame, so a seam needs a
// much broader low-frequency transition than the ownership antialiasing.
// These radii operate on the bounded 2400px analysis image.
const STREAMING_ILLUMINATION_LOCAL_SIGMA: f32 = 16.0;
const STREAMING_ILLUMINATION_CONTINUOUS_SIGMA: f32 = 640.0;
const STREAMING_ILLUMINATION_MIN_GAIN: f64 = 0.70;
const STREAMING_ILLUMINATION_MAX_GAIN: f64 = 1.40;
// The consensus pass runs on a bounded low-frequency grid.  It only supplies
// a smooth additive correction after source ownership is final; all brush and
// paper detail remains the pixel chosen by the ownership compositor.
// Keep this wider than the ownership antialiasing. Exposure blocks span
// hundreds of native pixels; a short kernel would only retouch the seam edge
// and leave the source rectangle intact. This is still low-frequency because
// the final correction is added to the untouched source-owned pixels.
const STREAMING_CONSENSUS_SIGMA: f32 = 64.0;
const STREAMING_CONSENSUS_MIN_SUPPORT: f32 = 0.001;
const STREAMING_CONSENSUS_MAX_DELTA: f64 = 0.18;
// Source-owner correction compares each owner's local low-frequency field
// with a much broader canvas reference.  It removes frame-sized exposure
// blocks while adding only a smooth field to source-owned pixels.
const STREAMING_OWNER_LOCAL_SIGMA: f32 = 64.0;
const STREAMING_OWNER_GLOBAL_SIGMA: f32 = 640.0;
// Flat-artwork source exposures can vary over a whole frame.  The overlap
// observations therefore need to influence the non-overlap interior over
// several 112px tone-field cells; a one-cell kernel leaves a source-sized
// exposure rectangle intact.  Focus-stack layers keep the narrower kernel.
const STREAMING_FLAT_TONE_SIGMA_CELLS: f64 = 16.0;
const GRID_STEP: u32 = 32;
const NATIVE_REFINE_STEP: u32 = 16;
const NATIVE_FIELD_RADIUS: f64 = 32.0;
// Selection cost added to a candidate that is sharper than the current
// composite but photometrically inconsistent with it (需求 3.3 requires ≥ 1.0).
pub(super) const OWNERSHIP_MISMATCH_PENALTY: f64 = 2.4;
const STREAMING_TILE_SIZE: u32 = 1024;
// A float RGB canvas costs twelve bytes per pixel before the ownership mask,
// decoded RAW layer, and canonical 16-bit result are counted.  Switch to the
// tiled backing store by canvas area as well as by capture-sequence metadata:
// evidence-only graph alignment intentionally has no synthetic sequence
// bridges, so using that flag alone made large, valid focus mosaics allocate
// the entire sparse bounding box in memory.
const STREAMING_CANVAS_MIN_PIXELS: u64 = 120_000_000;

fn luma(p: Rgb<f32>) -> f64 {
    f64::from(p[0] * 0.299 + p[1] * 0.587 + p[2] * 0.114)
}

fn rgb_at(image: &Rgb32FImage, x: f64, y: f64) -> Option<Rgb<f32>> {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= image.width() as f64
        || y >= image.height() as f64
    {
        return None;
    }
    Some(get_high_quality_interpolated_pixel(image, x, y))
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Register one non-anchor frame of a Capture_Station against the station's
/// already composited canvas at native resolution (需求 2.2–2.9).
///
/// This is the entry point of the *default* station path.  It runs the same
/// measurement as the mosaic comparison path — [`refine_layer`] twice at the
/// analysis resolution of 需求 2.2, then [`refine_native_layer`] with the
/// control point gates of 需求 2.3–2.8 — over a [`LayerSampler`] built from the
/// frame's own source pixels, and hands back both the report record and the
/// displacement field, so the caller's already rendered layer can be corrected
/// by exactly what was measured.
///
/// `layer` is the frame's canvas footprint as the caller rendered it, and
/// `transform` must be the pose that produced it: the sampler reproduces that
/// geometry, which is what makes the measured displacement applicable to those
/// pixels.
#[allow(clippy::too_many_arguments)]
pub(crate) fn register_station_frame_native(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    info: &ImageInfo,
    source: &Rgb32FImage,
    transform: &Matrix3<f64>,
    projection: Projection,
    offset: (f64, f64),
    layer: (u32, u32, u32, u32),
    anchor_long_side: u32,
    into_anchor: Matrix3<f64>,
) -> Option<StationFrameRegistration> {
    register_station_frame_native_with_mapping(
        base,
        base_mask,
        info,
        source,
        transform,
        projection,
        offset,
        layer,
        anchor_long_side,
        into_anchor,
        None,
    )
}

fn adjusted(pixel: Rgb<f32>, delta: [f64; 3]) -> Rgb<f32> {
    Rgb(std::array::from_fn(|c| {
        (pixel[c] * delta[c].exp() as f32).clamp(0.0, 1.0)
    }))
}

/// Test-only access to the Intra_Station_Registrar control point internals of
/// 需求 2.4 to 2.8.
///
/// Property 6 lives in `super::stack_pipeline::properties`, a sibling module that
/// cannot see `LayerSampler`, `NativeRefinePass` or `measure_native_control_point`.
/// The two drivers below assemble exactly what the in-memory compositor assembles
/// — a native-resolution base with a full coverage mask, a layer sampler at
/// `scale = 1.0` and an identity residual field — and then call the production
/// functions unchanged, so the property tests the refinement rather than a model
/// of it.
#[cfg(test)]
pub(crate) mod intra_station_test_access {
    use super::*;

    /// Everything one control point measured, plus the verdict the production
    /// pass gave it.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct ControlPointProbe {
        /// Position in the refinement grid, row major.
        pub(crate) column: usize,
        pub(crate) row: usize,
        /// The cell centre lies on covered base (需求 2.8).
        pub(crate) covered: bool,
        /// The patch pair was complete, so a match was attempted.
        pub(crate) evaluated: bool,
        /// Displacement of a completed bidirectional match, in native pixels of
        /// the anchor frame, before the 需求 2.4 / 2.6 gates.  `None` when the
        /// matcher's similarity gates or the reverse match stopped the
        /// measurement, i.e. when there is no located match to judge.
        pub(crate) displacement_px: Option<[f64; 2]>,
        /// Bidirectional round trip of that match, in native pixels (需求 2.4).
        pub(crate) round_trip_px: f64,
        /// Symmetric reprojection error of the global model there, in native
        /// pixels (需求 2.6).
        pub(crate) symmetric_error_px: f64,
        /// Forward peak correlation, `None` when no hypothesis was scored.
        pub(crate) best_correlation: Option<f64>,
        /// The displacement the production pass accepted at this position, i.e.
        /// what the 需求 2.5 neighbourhood gate let through.
        pub(crate) accepted_px: Option<[f64; 2]>,
    }

    /// One native refinement pass, measured control point by control point.
    ///
    /// Returns every grid position together with the accepted set of the real
    /// pass and the observation positions that entered the displacement field.
    pub(crate) fn probe_native_pass(
        base: &Rgb32FImage,
        layer: &Rgb32FImage,
        anchor_long_side: u32,
        spacing: f64,
    ) -> (Vec<ControlPointProbe>, Vec<(usize, usize)>, usize) {
        let (width, height) = base.dimensions();
        let info = image_info_for_probe(layer);
        let mask = GrayImage::from_pixel(width, height, Luma([255]));
        let sampler = LayerSampler {
            info: &info,
            source: layer,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            native_mapping: None,
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(width, height, GRID_STEP as f64),
        };
        let mode = NativeRefineMode::IntraStation { anchor_long_side };
        let window = NativeRefineBase {
            image: base,
            mask: &mask,
            left: 0,
            top: 0,
        };
        let mut pass = NativeRefinePass::new(width, height);
        collect_native_observations(&window, &sampler, width, height, spacing, mode, &mut pass);

        let (match_radius, search_samples) = mode.patch_geometry();
        let columns = (20..width.saturating_sub(20))
            .step_by(NATIVE_REFINE_STEP as usize)
            .collect::<Vec<_>>();
        let rows = (20..height.saturating_sub(20))
            .step_by(NATIVE_REFINE_STEP as usize)
            .collect::<Vec<_>>();
        let mut probes = Vec::with_capacity(columns.len() * rows.len());
        for (row, &y) in rows.iter().enumerate() {
            for (column, &x) in columns.iter().enumerate() {
                // `collect_native_observations` took `&sampler`, so the sampler is
                // in the state it measured with and re-measuring one control point
                // reproduces its own measurement exactly.
                let measurement = measure_native_control_point(
                    &window,
                    &sampler,
                    x,
                    y,
                    spacing,
                    mode,
                    match_radius,
                    search_samples,
                );
                probes.push(ControlPointProbe {
                    column,
                    row,
                    covered: measurement.covered,
                    evaluated: measurement.evaluated,
                    displacement_px: measurement.measured.map(|(d, _, _, _)| d),
                    round_trip_px: measurement.round_trip_px,
                    symmetric_error_px: measurement
                        .measured
                        .map(|(_, _, _, error)| error)
                        .unwrap_or(0.0),
                    best_correlation: measurement.best_correlation,
                    accepted_px: pass.accepted_grid.accepted_at(column, row),
                });
            }
        }
        let observations = pass
            .observations
            .iter()
            .map(|(position, _, _)| {
                let column = columns
                    .iter()
                    .position(|&x| f64::from(x) == position.x)
                    .expect("an observation sits on a grid column");
                let row = rows
                    .iter()
                    .position(|&y| f64::from(y) == position.y)
                    .expect("an observation sits on a grid row");
                (column, row)
            })
            .collect();
        (probes, observations, pass.evaluated)
    }

    /// What a whole native refinement decided about one frame (需求 2.7 / 2.8).
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct RefinementProbe {
        /// Accepted control point cells over *both* spacing passes, which is the
        /// numerator 需求 2.8 defines.
        pub(crate) accepted_control_points: usize,
        /// Accepted control point cells of the final spacing pass alone.
        pub(crate) last_pass_accepted_control_points: usize,
        /// Control point cells whose centre lay on covered base in the final
        /// pass: the denominator of 需求 2.8, in cells.
        pub(crate) valid_cells: usize,
        /// Area of one control point cell, in native pixels of the anchor frame.
        pub(crate) cell_area_px: f64,
        pub(crate) evaluated_control_points: usize,
        pub(crate) local_baseline_samples: usize,
        pub(crate) global_baseline_samples: usize,
        /// Median symmetric reprojection error after the refinement, in native
        /// pixels (需求 2.7).
        pub(crate) local_median_symmetric_error_px: f64,
        /// The same median for the global model (需求 2.7).
        pub(crate) global_median_symmetric_error_px: f64,
        pub(crate) local_field_verdict: intra_station::LocalFieldVerdict,
        pub(crate) reverted_to_global: bool,
        pub(crate) inlier_area_coverage: f64,
        pub(crate) status: IntraStationFrameStatus,
        /// `true` when the displacement field the sampler carries afterwards is
        /// bit for bit the one it entered with, i.e. every control point position
        /// kept the global model's displacement.
        pub(crate) field_unchanged: bool,
    }

    /// Drive the whole native refinement of one frame (both spacing passes).
    pub(crate) fn probe_native_refinement(
        base: &Rgb32FImage,
        layer: &Rgb32FImage,
        anchor_long_side: u32,
    ) -> RefinementProbe {
        let (width, height) = base.dimensions();
        let info = image_info_for_probe(layer);
        let mask = GrayImage::from_pixel(width, height, Luma([255]));
        let entry = Field::new(width, height, GRID_STEP as f64);
        let mut sampler = LayerSampler {
            info: &info,
            source: layer,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            native_mapping: None,
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: entry.clone(),
        };
        let outcome = refine_native_layer(
            base,
            &mask,
            &mut sampler,
            width,
            height,
            NativeRefineMode::IntraStation { anchor_long_side },
        );
        let field_unchanged = sampler.residual.values.len() == entry.values.len()
            && sampler
                .residual
                .values
                .iter()
                .zip(entry.values.iter())
                .all(|(left, right)| {
                    left[0].to_bits() == right[0].to_bits()
                        && left[1].to_bits() == right[1].to_bits()
                });
        RefinementProbe {
            accepted_control_points: outcome.accepted_control_points,
            last_pass_accepted_control_points: outcome.last_pass_accepted_control_points,
            valid_cells: outcome.valid_cells,
            cell_area_px: outcome.cell_area_px,
            evaluated_control_points: outcome.evaluated_control_points,
            local_baseline_samples: outcome.local_baseline_samples,
            global_baseline_samples: outcome.global_baseline_samples,
            local_median_symmetric_error_px: outcome.local_median_symmetric_error_px,
            global_median_symmetric_error_px: outcome.global_median_symmetric_error_px,
            local_field_verdict: outcome.local_field_verdict,
            reverted_to_global: outcome.reverted_to_global,
            inlier_area_coverage: outcome.inlier_area_coverage,
            status: outcome.frame_status(),
            field_unchanged,
        }
    }

    fn image_info_for_probe(image: &Rgb32FImage) -> ImageInfo {
        ImageInfo {
            id: 0,
            filename: "/tmp/stack-pipeline/intra-station-probe.NEF".to_string(),
            width: image.width(),
            height: image.height(),
            alignment_image: image::DynamicImage::ImageRgb32F(image.clone()).to_luma8(),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm: Some(75.0),
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_control_cell_can_find_a_corner_missed_by_the_regular_probe_phase() {
        let image = Rgb32FImage::from_fn(128, 128, |x, y| {
            Rgb([if x >= 56 && y >= 56 { 0.9 } else { 0.1 }; 3])
        });
        let mask = GrayImage::from_pixel(128, 128, Luma([255]));
        let info = image_info(0, &image);
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
            residual: Field::new(128, 128, 48.0),
        };
        let base = NativeRefineBase {
            image: &image,
            mask: &mask,
            left: 0,
            top: 0,
        };
        assert_eq!(
            contrasted_native_control_location(&base, &sampler, 52, 52, 128, 128),
            (56, 56)
        );
        let uncovered = GrayImage::new(128, 128);
        let base = NativeRefineBase {
            mask: &uncovered,
            ..base
        };
        assert_eq!(
            contrasted_native_control_location(&base, &sampler, 52, 52, 128, 128),
            (52, 52)
        );
    }

    #[test]
    fn only_near_black_pixels_in_the_lower_left_capture_border_are_rejected() {
        let image = Rgb32FImage::new(100, 100);
        assert!(is_lower_left_black_capture_border(
            &image,
            10.0,
            90.0,
            Rgb([0.005, 0.006, 0.004]),
        ));
        assert!(!is_lower_left_black_capture_border(
            &image,
            50.0,
            90.0,
            Rgb([0.005, 0.006, 0.004]),
        ));
        assert!(!is_lower_left_black_capture_border(
            &image,
            10.0,
            90.0,
            Rgb([0.08, 0.03, 0.02]),
        ));
        // A pure-black lower-left sample remains non-photographic even at the
        // extreme source edge. Keeping it created the reported black block.
        assert!(is_lower_left_black_capture_border(
            &image,
            10.0,
            98.0,
            Rgb([0.005, 0.006, 0.004]),
        ));
    }

    #[test]
    fn streaming_store_round_trips_across_tile_boundaries() {
        let store = StreamingMosaicStore::new(2050, 1025).expect("tile store should initialize");
        for row in 0..store.tile_rows {
            for column in 0..store.tile_columns {
                let (_, _, width, height) = store.tile_extent(column, row);
                let value = (column + row * 3) as f32;
                let rgb =
                    Rgb32FImage::from_pixel(width, height, Rgb([value, value + 1.0, value + 2.0]));
                let mask = GrayImage::from_pixel(width, height, Luma([255]));
                store
                    .save_tile(column, row, &rgb, &mask)
                    .expect("tile should be writable");
            }
        }
        let crop = store
            .covered_bounds()
            .expect("coverage should be readable")
            .expect("full tile fixture should be covered");
        assert_eq!(crop, (0, 0, 2050, 1025));
        let output = store.materialize(crop).expect("tiles should materialize");
        assert_eq!(output.dimensions(), (2050, 1025));
        assert_eq!(*output.get_pixel(1023, 10), Rgb([0.0, 1.0, 2.0]));
        assert_eq!(*output.get_pixel(1024, 10), Rgb([1.0, 2.0, 3.0]));
        assert_eq!(*output.get_pixel(10, 1024), Rgb([3.0, 4.0, 5.0]));
    }

    #[test]
    fn covered_bounds_preserve_non_rectangular_source_edges() {
        let mask = GrayImage::from_fn(8, 6, |x, y| {
            Luma([u8::from((x == 1 && y == 3) || (x >= 3 && x <= 6 && y >= 1 && y <= 4)) * 255])
        });
        assert_eq!(mask_covered_bounds(&mask), Some((1, 1, 6, 4)));
    }

    #[test]
    fn streaming_covered_bounds_do_not_reduce_to_largest_rectangle() {
        let store = StreamingMosaicStore::new(12, 8).expect("tile store should initialize");
        let rgb = Rgb32FImage::new(12, 8);
        let mask = GrayImage::from_fn(12, 8, |x, y| {
            Luma([u8::from((x == 1 && y == 6) || (x >= 4 && x <= 10 && y >= 1 && y <= 5)) * 255])
        });
        store
            .save_tile(0, 0, &rgb, &mask)
            .expect("tile should be writable");
        assert_eq!(
            store.covered_bounds().expect("coverage should be readable"),
            Some((1, 1, 10, 6))
        );
    }

    #[test]
    fn streaming_decision_alpha_interpolates_between_cells() {
        let decision =
            GrayImage::from_fn(2, 2, |x, y| Luma([if (x + y) % 2 == 0 { 0 } else { 255 }]));
        let centre = sample_decision_alpha(&decision, 0.5, 0.5);
        assert!((centre - 0.5).abs() < 1e-6);
        let left = sample_decision_alpha(&decision, 0.05, 0.5);
        let right = sample_decision_alpha(&decision, 0.95, 0.5);
        assert!(left > 0.45 && left < 0.55);
        assert!(right > 0.45 && right < 0.55);
        assert!(sample_decision_alpha(&decision, 0.0, 0.0) < 1e-6);
        assert!((sample_decision_alpha(&decision, 1.0, 0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn streaming_materialize_preserves_a_natural_gradient_at_owner_boundary() {
        let store = StreamingMosaicStore::new(256, 32).expect("tile store should initialize");
        let rgb = Rgb32FImage::from_fn(256, 32, |x, _| {
            let value = 0.30 + 0.40 * x as f32 / 255.0;
            Rgb([value, value * 0.98, value * 0.94])
        });
        let mask = GrayImage::from_pixel(256, 32, Luma([255]));
        let owner = GrayImage::from_fn(256, 32, |x, _| Luma([if x < 128 { 1 } else { 2 }]));
        store
            .save_tile(0, 0, &rgb, &mask)
            .expect("tile should be writable");
        store
            .save_owner_tile(0, 0, &owner)
            .expect("owner should be writable");
        let output = materialize_streaming_focus_output(&store, (0, 0, 256, 32))
            .expect("gradient should materialize");
        let left = luma(*output.get_pixel(126, 16));
        let right = luma(*output.get_pixel(129, 16));
        let expected = luma(*rgb.get_pixel(126, 16)) - luma(*rgb.get_pixel(129, 16));
        assert!((left - right).abs() < expected.abs() * 1.4);
        assert!((left - luma(*rgb.get_pixel(126, 16))).abs() < 1e-6);
        assert!((right - luma(*rgb.get_pixel(129, 16))).abs() < 1e-6);
    }

    #[test]
    fn streaming_analysis_region_is_continuous_across_tile_boundaries() {
        let store = StreamingMosaicStore::new(2050, 32).expect("tile store should initialize");
        for column in 0..store.tile_columns {
            let (_, _, width, height) = store.tile_extent(column, 0);
            let value = column as f32 + 1.0;
            let rgb = Rgb32FImage::from_pixel(width, height, Rgb([value, value, value]));
            let mask = GrayImage::from_pixel(width, height, Luma([255]));
            store
                .save_tile(column, 0, &rgb, &mask)
                .expect("tile should be writable");
        }

        let (analysis, mask, owner) = store
            .analysis_region(1018, 4, 2.0, 8, 4)
            .expect("analysis region should be readable");
        assert_eq!(analysis.dimensions(), (8, 4));
        assert_eq!(*analysis.get_pixel(2, 1), Rgb([1.0, 1.0, 1.0]));
        assert_eq!(*analysis.get_pixel(3, 1), Rgb([2.0, 2.0, 2.0]));
        assert!(mask.as_raw().iter().all(|&value| value == 255));
        assert!(owner.as_raw().iter().all(|&value| value == 0));
    }

    #[test]
    fn low_frequency_harmonization_softens_group_exposure_steps_without_blurring_detail() {
        let store = StreamingMosaicStore::new(256, 64).expect("tile store should initialize");
        let rgb = Rgb32FImage::from_fn(256, 64, |x, y| {
            let base = if x < 128 { 0.42 } else { 0.58 };
            let detail = if (x / 4 + y / 4) % 2 == 0 {
                0.03
            } else {
                -0.03
            };
            Rgb([base + detail, base + detail, base + detail])
        });
        let mask = GrayImage::from_pixel(256, 64, Luma([255]));
        let owner = GrayImage::from_fn(256, 64, |x, _| Luma([if x < 128 { 1 } else { 2 }]));
        store
            .save_tile(0, 0, &rgb, &mask)
            .expect("tile should be writable");
        store
            .save_owner_tile(0, 0, &owner)
            .expect("owner should be writable");
        let (gains, scale) = streaming_seam_harmonization(&store, (0, 0, 256, 64), &[])
            .expect("harmonization should build")
            .expect("two owners should create a seam field");
        let mut output = rgb.clone();
        apply_streaming_seam_harmonization(&mut output, &gains, scale);
        let band_mean = |image: &Rgb32FImage, start: u32| {
            (start..start + 8)
                .map(|x| luma(*image.get_pixel(x, 32)))
                .sum::<f64>()
                / 8.0
        };
        let before_step = (band_mean(&rgb, 120) - band_mean(&rgb, 128)).abs();
        let after_step = (band_mean(&output, 120) - band_mean(&output, 128)).abs();
        assert!(
            after_step < before_step * 0.72,
            "the exposure seam should soften: {before_step:.4} -> {after_step:.4}"
        );
        let before_detail = (luma(*rgb.get_pixel(20, 20)) - luma(*rgb.get_pixel(24, 20))).abs();
        let after_detail =
            (luma(*output.get_pixel(20, 20)) - luma(*output.get_pixel(24, 20))).abs();
        assert!(
            after_detail >= before_detail * 0.90,
            "multiplicative low-frequency correction must retain local detail"
        );
    }

    #[test]
    fn source_owner_low_frequency_correction_reduces_frame_step_and_keeps_detail() {
        let store = StreamingMosaicStore::new(256, 64).expect("tile store should initialize");
        let rgb = Rgb32FImage::from_fn(256, 64, |x, y| {
            let base = if x < 128 { 0.40 } else { 0.60 };
            let detail = if (x / 3 + y / 3) % 2 == 0 {
                0.025
            } else {
                -0.025
            };
            Rgb([base + detail, base + detail * 0.9, base + detail * 0.8])
        });
        let mask = GrayImage::from_pixel(256, 64, Luma([255]));
        let owner = GrayImage::from_fn(256, 64, |x, _| Luma([if x < 128 { 1 } else { 2 }]));
        store
            .save_tile(0, 0, &rgb, &mask)
            .expect("tile should be writable");
        store
            .save_owner_tile(0, 0, &owner)
            .expect("owner should be writable");
        let mut output = rgb.clone();
        let before_step =
            (luma(*output.get_pixel(120, 32)) - luma(*output.get_pixel(136, 32))).abs();
        let before_detail =
            (luma(*output.get_pixel(30, 30)) - luma(*output.get_pixel(33, 30))).abs();
        let corrected = apply_streaming_source_owner_low_frequency_correction(
            &store,
            (0, 0, 256, 64),
            &mut output,
        )
        .expect("owner correction should run");
        let after_step =
            (luma(*output.get_pixel(120, 32)) - luma(*output.get_pixel(136, 32))).abs();
        let after_detail =
            (luma(*output.get_pixel(30, 30)) - luma(*output.get_pixel(33, 30))).abs();
        assert!(
            corrected > 0,
            "owner fields should contribute correction cells"
        );
        assert!(
            after_step < before_step * 0.72,
            "source exposure step should shrink: {before_step:.4} -> {after_step:.4}"
        );
        assert!(
            after_detail >= before_detail * 0.88,
            "owner correction must not erase source detail: {before_detail:.4} -> {after_detail:.4}"
        );
    }

    #[test]
    fn source_owner_correction_reaches_the_interior_of_a_large_source_block() {
        let (width, height) = (2048, 64);
        let store = StreamingMosaicStore::new(width, height).expect("tile store should initialize");
        let rgb = Rgb32FImage::from_fn(width, height, |x, y| {
            let base = if x < width / 2 { 0.32 } else { 0.58 };
            let detail = if (x / 5 + y / 5) % 2 == 0 {
                0.02
            } else {
                -0.02
            };
            Rgb([base + detail, base + detail * 0.95, base + detail * 0.9])
        });
        let mask = GrayImage::from_pixel(width, height, Luma([255]));
        let owner = GrayImage::from_fn(width, height, |x, _| {
            Luma([if x < width / 2 { 1 } else { 2 }])
        });
        for column in 0..2 {
            let left = column * 1024;
            let rgb_tile = image::imageops::crop_imm(&rgb, left, 0, 1024, height).to_image();
            let mask_tile = image::imageops::crop_imm(&mask, left, 0, 1024, height).to_image();
            let owner_tile = image::imageops::crop_imm(&owner, left, 0, 1024, height).to_image();
            store
                .save_tile(column, 0, &rgb_tile, &mask_tile)
                .expect("tile should be writable");
            store
                .save_owner_tile(column, 0, &owner_tile)
                .expect("owner should be writable");
        }
        let mut output = rgb.clone();
        let before = (luma(*output.get_pixel(256, 32)) - luma(*output.get_pixel(1792, 32))).abs();
        let detail_before =
            (luma(*output.get_pixel(500, 32)) - luma(*output.get_pixel(505, 32))).abs();
        apply_streaming_source_owner_low_frequency_correction(
            &store,
            (0, 0, width, height),
            &mut output,
        )
        .expect("owner correction should run");
        let after = (luma(*output.get_pixel(256, 32)) - luma(*output.get_pixel(1792, 32))).abs();
        let detail_after =
            (luma(*output.get_pixel(500, 32)) - luma(*output.get_pixel(505, 32))).abs();
        assert!(
            after < before * 0.72,
            "large source step should shrink: {before:.4} -> {after:.4}"
        );
        assert!(
            detail_after >= detail_before * 0.88,
            "large source correction must retain detail"
        );
    }

    #[test]
    fn streaming_consensus_adds_supported_low_frequency_delta_without_erasing_detail() {
        let width = 96;
        let height = 64;
        let base = Rgb32FImage::from_fn(width, height, |x, y| {
            let detail = if (x / 3 + y / 3) % 2 == 0 {
                0.025
            } else {
                -0.025
            };
            Rgb([0.42 + detail, 0.39 + detail, 0.34 + detail])
        });
        let candidate = Rgb32FImage::from_fn(width, height, |x, y| {
            let detail = if (x / 3 + y / 3) % 2 == 0 {
                0.025
            } else {
                -0.025
            };
            Rgb([0.52 + detail, 0.49 + detail, 0.44 + detail])
        });
        let mask = GrayImage::from_pixel(width, height, Luma([255]));
        let tone = Field::new(width, height, 112.0);
        let mut consensus = StreamingLowFrequencyConsensus::new(width, height);
        consensus.observe_overlap(0, 0, 1.0, &base, &mask, &candidate, &mask, &tone);
        let mut output = base.clone();
        let before_detail =
            (luma(*output.get_pixel(30, 30)) - luma(*output.get_pixel(33, 30))).abs();
        consensus.apply(&mut output, (0, 0, width, height));
        let after_detail =
            (luma(*output.get_pixel(30, 30)) - luma(*output.get_pixel(33, 30))).abs();
        let before_mean = luma(*base.get_pixel(48, 32));
        let after_mean = luma(*output.get_pixel(48, 32));
        assert!(
            after_mean > before_mean + 0.06,
            "consensus correction was not applied"
        );
        assert!(
            after_detail >= before_detail * 0.85,
            "low-frequency correction erased source detail: {before_detail} -> {after_detail}"
        );
    }

    #[test]
    fn streaming_consensus_does_not_reapply_the_raw_candidate_exposure() {
        let width = 96;
        let height = 64;
        let base = Rgb32FImage::from_pixel(width, height, Rgb([0.60, 0.60, 0.60]));
        let candidate = Rgb32FImage::from_pixel(width, height, Rgb([0.30, 0.30, 0.30]));
        let mask = GrayImage::from_pixel(width, height, Luma([255]));
        let mut tone = Field::new(width, height, 112.0);
        tone.values.fill([2.0f64.ln(); 3]);
        let mut consensus = StreamingLowFrequencyConsensus::new(width, height);
        consensus.observe_overlap(0, 0, 1.0, &base, &mask, &candidate, &mask, &tone);
        let mut output = base.clone();
        consensus.apply(&mut output, (0, 0, width, height));
        let mut residual = 0.0f64;
        for y in 0..height {
            for x in 0..width {
                residual = residual
                    .max((luma(*output.get_pixel(x, y)) - luma(*base.get_pixel(x, y))).abs());
            }
        }
        assert!(
            residual < 0.01,
            "consensus reapplied raw exposure after tone correction: {residual:.4}"
        );
    }

    #[test]
    fn streaming_consensus_propagates_an_overlap_correction_into_a_source_block() {
        let width = 256;
        let height = 64;
        let base = Rgb32FImage::from_fn(width, height, |x, _| {
            let value = if x < 128 { 0.40 } else { 0.60 };
            Rgb([value, value, value])
        });
        let candidate = Rgb32FImage::from_pixel(width, height, Rgb([0.60, 0.60, 0.60]));
        let overlap = GrayImage::from_fn(width, height, |x, _| {
            Luma([u8::from((112..=144).contains(&x)) * 255])
        });
        let candidate_mask = GrayImage::from_pixel(width, height, Luma([255]));
        let tone = Field::new(width, height, 112.0);
        let mut consensus = StreamingLowFrequencyConsensus::new(width, height);
        consensus.observe_overlap(
            0,
            0,
            1.0,
            &base,
            &overlap,
            &candidate,
            &candidate_mask,
            &tone,
        );
        let mut output = base.clone();
        consensus.apply(&mut output, (0, 0, width, height));
        assert!(
            luma(*output.get_pixel(32, 32)) > 0.53,
            "the overlap correction should reach the low-exposure source block"
        );
        assert!(
            (luma(*output.get_pixel(32, 32)) - luma(*output.get_pixel(224, 32))).abs() < 0.08,
            "the source exposure step should be reduced across the block"
        );
    }

    #[test]
    fn frequency_separation_removes_station_blocks_but_keeps_detail_and_black_bounds() {
        let mut image = Rgb32FImage::from_fn(512, 128, |x, y| {
            if y < 8 {
                return Rgb([0.0; 3]);
            }
            let base = if x < 256 { 0.36 } else { 0.54 };
            let detail = if (x / 3 + y / 3) % 2 == 0 {
                0.035
            } else {
                -0.035
            };
            Rgb([base + detail, base + detail, base + detail])
        });
        let original = image.clone();
        smooth_streaming_low_frequency_illumination(&mut image);
        let mean = |source: &Rgb32FImage, start: u32| {
            (start..start + 12)
                .map(|x| luma(*source.get_pixel(x, 64)))
                .sum::<f64>()
                / 12.0
        };
        let before = (mean(&original, 238) - mean(&original, 262)).abs();
        let after = (mean(&image, 238) - mean(&image, 262)).abs();
        assert!(
            after < before * 0.82,
            "station step {before:.4} -> {after:.4}"
        );
        let before_detail =
            (luma(*original.get_pixel(80, 64)) - luma(*original.get_pixel(83, 64))).abs();
        let after_detail = (luma(*image.get_pixel(80, 64)) - luma(*image.get_pixel(83, 64))).abs();
        assert!(after_detail >= before_detail * 0.88);
        assert!((0..512).all(|x| image.get_pixel(x, 2).0 == [0.0; 3]));
    }

    #[test]
    fn same_coordinate_group_relations_recover_exposure_offset() {
        let (width, height) = (320, 160);
        let base = Rgb32FImage::from_fn(width, height, |x, y| {
            let detail = if (x / 5 + y / 5) % 2 == 0 {
                0.025
            } else {
                -0.025
            };
            Rgb([0.42 + detail, 0.39 + detail, 0.34 + detail])
        });
        let expected = [0.12f64, 0.08, -0.03];
        let candidate = Rgb32FImage::from_fn(width, height, |x, y| {
            let source = base.get_pixel(x, y);
            Rgb(std::array::from_fn(|channel| {
                source[channel] * expected[channel].exp() as f32
            }))
        });
        let mask = GrayImage::from_pixel(width, height, Luma([255]));
        let owner = GrayImage::from_pixel(width, height, Luma([1]));
        let tone = Field::new(width, height, 112.0);
        let relations = streaming_group_tone_relations_from_analysis(
            &base, &mask, &owner, &candidate, &mask, &tone, 2,
        );
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].0, (1, 2));
        for (actual, expected) in relations[0].1.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 0.01, "{actual} vs {expected}");
        }
    }

    #[test]
    #[ignore = "applies production low-frequency correction to a supplied real preview"]
    fn real_preview_low_frequency_correction_from_env() {
        let input = std::env::var_os("RAW_EDITOR_MOSAIC_PREVIEW_INPUT")
            .expect("RAW_EDITOR_MOSAIC_PREVIEW_INPUT is required");
        let output = std::env::var_os("RAW_EDITOR_MOSAIC_PREVIEW_OUTPUT")
            .expect("RAW_EDITOR_MOSAIC_PREVIEW_OUTPUT is required");
        let mut image = image::open(input)
            .expect("real preview should open")
            .to_rgb32f();
        smooth_streaming_low_frequency_illumination(&mut image);
        image::DynamicImage::ImageRgb32F(image)
            .to_rgb8()
            .save(output)
            .expect("corrected preview should save");
    }

    #[test]
    fn group_exposure_harmonization_cannot_draw_a_halo_around_dark_strokes() {
        let store = StreamingMosaicStore::new(256, 64).expect("tile store should initialize");
        let rgb = Rgb32FImage::from_fn(256, 64, |x, _| {
            let paper = if x < 128 { 0.46 } else { 0.56 };
            let value = if (58..=62).contains(&x) { 0.06 } else { paper };
            Rgb([value, value * 0.96, value * 0.90])
        });
        let mask = GrayImage::from_pixel(256, 64, Luma([255]));
        let owner = GrayImage::from_fn(256, 64, |x, _| Luma([if x < 128 { 1 } else { 2 }]));
        store
            .save_tile(0, 0, &rgb, &mask)
            .expect("tile should be writable");
        store
            .save_owner_tile(0, 0, &owner)
            .expect("owner should be writable");
        let (gains, _) = streaming_seam_harmonization(&store, (0, 0, 256, 64), &[])
            .expect("harmonization should build")
            .expect("two owners should create a gain field");
        let paper_gain = gains.get_pixel(30, 32)[0];
        for x in [52, 56, 58, 60, 62, 64, 68] {
            assert!(
                (gains.get_pixel(x, 32)[0] - paper_gain).abs() < 1e-5,
                "a dark stroke must not alter the exposure gain around itself"
            );
        }
    }

    #[test]
    fn streaming_uncovered_fill_preserves_owner_and_closes_real_source_holes() {
        let source = Rgb32FImage::from_pixel(16, 8, Rgb([0.8, 0.7, 0.6]));
        let info = image_info(1, &source);
        let mut tile = Rgb32FImage::from_pixel(16, 8, Rgb([0.2, 0.2, 0.2]));
        let mut mask = GrayImage::from_pixel(16, 8, Luma([255]));
        let mut owner = GrayImage::from_pixel(16, 8, Luma([3]));
        mask.put_pixel(5, 3, Luma([0]));
        owner.put_pixel(5, 3, Luma([0]));
        let sampler = LayerSampler {
            info: &info,
            source: &source,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            native_mapping: None,
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(16, 8, 32.0),
        };
        fill_streaming_uncovered_tile_region(
            &mut tile,
            &mut mask,
            &mut owner,
            4,
            &sampler,
            &Field::new(16, 8, 32.0),
            0,
            0,
            0,
            15,
            0,
            7,
            1.0,
        );

        assert_eq!(*tile.get_pixel(4, 3), Rgb([0.2, 0.2, 0.2]));
        assert_eq!(*tile.get_pixel(5, 3), Rgb([0.8, 0.7, 0.6]));
        assert_eq!(mask.get_pixel(5, 3)[0], 255);
        assert_eq!(owner.get_pixel(4, 3)[0], 3);
        assert_eq!(owner.get_pixel(5, 3)[0], 4);
    }

    #[test]
    fn focus_group_ownership_cannot_reclaim_another_camera_position() {
        let sharp = sharp_fixture();
        let base = image::imageops::blur(&sharp, 2.0);
        let mask = GrayImage::from_pixel(sharp.width(), sharp.height(), Luma([255]));
        let owner = GrayImage::from_fn(sharp.width(), sharp.height(), |x, _| {
            Luma([if x < sharp.width() / 2 { 3 } else { 4 }])
        });
        let decision = streaming_ownership_from_analysis(
            &base,
            &mask,
            &sharp,
            &mask,
            &Field::new(sharp.width(), sharp.height(), 32.0),
            &owner,
            3,
            false,
        );
        let left_selected = decision
            .enumerate_pixels()
            .filter(|(x, _, pixel)| *x < decision.width() / 2 && pixel[0] != 0)
            .count();
        let right_selected = decision
            .enumerate_pixels()
            .filter(|(x, _, pixel)| *x >= decision.width() / 2 && pixel[0] != 0)
            .count();
        assert!(
            left_selected > 0,
            "the sharper member should contribute inside its group"
        );
        assert_eq!(
            right_selected, 0,
            "a focal member must not overwrite a different camera position"
        );
    }

    #[test]
    fn panorama_group_transition_only_enters_from_new_coverage() {
        let candidate = sharp_fixture();
        let base = image::imageops::blur(&candidate, 1.0);
        let mut base_mask = GrayImage::new(candidate.width(), candidate.height());
        let mut owner = GrayImage::new(candidate.width(), candidate.height());
        for y in 0..candidate.height() {
            for x in 0..candidate.width() * 3 / 4 {
                base_mask.put_pixel(x, y, Luma([255]));
                owner.put_pixel(x, y, Luma([2]));
            }
        }
        let candidate_mask =
            GrayImage::from_pixel(candidate.width(), candidate.height(), Luma([255]));
        let decision = streaming_ownership_from_analysis(
            &base,
            &base_mask,
            &candidate,
            &candidate_mask,
            &Field::new(candidate.width(), candidate.height(), 32.0),
            &owner,
            3,
            true,
        );
        let left_quarter_selected = decision
            .enumerate_pixels()
            .filter(|(x, _, pixel)| *x < decision.width() / 4 && pixel[0] != 0)
            .count();
        let overlap_selected = decision
            .enumerate_pixels()
            .filter(|(x, _, pixel)| {
                *x >= decision.width() / 2 && *x < decision.width() * 3 / 4 && pixel[0] != 0
            })
            .count();
        let new_coverage_selected = decision
            .enumerate_pixels()
            .filter(|(x, _, pixel)| *x >= decision.width() * 3 / 4 && pixel[0] != 0)
            .count();
        assert_eq!(
            left_quarter_selected, 0,
            "a new camera group must not form a detached sharpness island"
        );
        assert!(
            new_coverage_selected > 0,
            "new source coverage must be admitted"
        );
        assert!(
            overlap_selected > 0,
            "a panorama seam must be allowed to leave the rectangular new-coverage boundary"
        );
    }

    #[test]
    fn panorama_transition_marks_ink_cells_as_expensive_seam_routes() {
        let image = Rgb32FImage::from_fn(96, 64, |x, y| {
            let ink = (38..58).contains(&x) && (18..46).contains(&y);
            let value = if ink { 0.05 } else { 0.52 };
            Rgb([value, value * 0.96, value * 0.90])
        });
        let quiet = streaming_transition_structure(|x, y| rgb_at(&image, x, y), 0.0, 0.0, 24.0);
        let ink = streaming_transition_structure(|x, y| rgb_at(&image, x, y), 30.0, 12.0, 36.0);
        assert!(quiet < 0.05, "plain paper should remain a cheap seam route");
        assert!(
            ink > 0.75,
            "a cell crossed by dark calligraphy must strongly repel the panorama seam"
        );
    }

    fn image_info(id: usize, image: &Rgb32FImage) -> ImageInfo {
        ImageInfo {
            id,
            filename: format!("source-{id}.png"),
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

    fn sharp_fixture() -> Rgb32FImage {
        Rgb32FImage::from_fn(160, 120, |x, y| {
            let value = 0.36
                + 0.12 * (x as f32 * 0.75).sin()
                + 0.10 * (y as f32 * 0.65).cos()
                + 0.03 * ((x + y) as f32 * 0.19).sin();
            Rgb([value * 1.1, value, value * 0.8])
        })
    }

    fn noisy_colour_stroke_pair() -> (Rgb32FImage, Rgb32FImage) {
        let sharp = Rgb32FImage::from_fn(192, 160, |x, y| {
            let stroke = (x as i32 - 96).abs() <= 3 && (20..140).contains(&y);
            // A faded red mark can have almost the same luminance as its
            // brown substrate; colour edges still carry genuine focus.
            if stroke {
                Rgb([0.46, 0.31, 0.27])
            } else {
                Rgb([0.40, 0.34, 0.27])
            }
        });
        let blurred = image::imageops::blur(&sharp, 3.2);
        let noisy = |im: &Rgb32FImage, amplitude: f32| {
            Rgb32FImage::from_fn(im.width(), im.height(), |x, y| {
                let hash = x.wrapping_mul(0x9e3779b9) ^ y.wrapping_mul(0x85ebca6b);
                let hash = (hash ^ (hash >> 16)).wrapping_mul(0x7feb352d);
                let noise = ((hash & 65535) as f32 / 65535.0 - 0.5) * amplitude;
                Rgb(im.get_pixel(x, y).0.map(|v| (v + noise).clamp(0.0, 1.0)))
            })
        };
        (noisy(&sharp, 0.020), noisy(&blurred, 0.080))
    }

    /// Aperiodic paper grain.  A sum of sines would be rejected by
    /// `refine_warped_patch`'s second-peak guard for looking like a repeated
    /// stroke, which is the guard doing its job rather than a refinement failure.
    fn refine_texture(width: u32, height: u32) -> Rgb32FImage {
        let grain = Rgb32FImage::from_fn(width, height, |x, y| {
            let hash = x.wrapping_mul(0x9e37_79b9) ^ y.wrapping_mul(0x85eb_ca6b);
            let hash = (hash ^ (hash >> 15)).wrapping_mul(0x7feb_352d);
            let value = 0.25 + 0.50 * ((hash >> 8) & 0xffff) as f32 / 65_535.0;
            Rgb([value, value * 0.96, value * 0.90])
        });
        image::imageops::blur(&grain, 1.1)
    }

    /// The intra-station gates must leave usable control points.
    ///
    /// Without this the native refinement could reject every control point and
    /// still look healthy from the outside: the frame simply reverts to its
    /// global model, the output is byte identical to a run with no refinement at
    /// all, and nothing fails.  That is exactly what a matching window radius of
    /// 48 inside a 57×57 patch buffer does — `refine_warped_patch` needs a sample
    /// at −20 and returns `None` for every position.
    #[test]
    fn intra_station_native_refinement_accepts_control_points() {
        let base = refine_texture(256, 256);
        // The layer is the same photograph displaced by a few native pixels,
        // which is what focus breathing does inside one Capture_Station.
        // Two native pixels: inside 需求 2.6's `0.01 × 256` error ceiling for this
        // fixture, so the gate under test is the matching, not the ceiling.
        let shifted = Rgb32FImage::from_fn(256, 256, |x, y| *base.get_pixel((x + 2).min(255), y));
        let info = image_info(1, &shifted);
        let mask = GrayImage::from_pixel(256, 256, Luma([255]));
        let mut sampler = LayerSampler {
            info: &info,
            source: &shifted,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            native_mapping: None,
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(256, 256, GRID_STEP as f64),
        };
        let outcome = refine_native_layer(
            &base,
            &mask,
            &mut sampler,
            256,
            256,
            NativeRefineMode::IntraStation {
                anchor_long_side: 256,
            },
        );
        assert!(
            !outcome.skipped,
            "需求 2.2 has no analysis-scale early exit"
        );
        assert!(
            outcome.evaluated_control_points >= 64,
            "the control point grid must reach the layer ({} evaluated)",
            outcome.evaluated_control_points
        );
        assert!(
            outcome.accepted_control_points >= 16,
            "the 需求 2.4/2.5/2.6 gates must accept real matches, not reject every one ({}/{} accepted)",
            outcome.accepted_control_points,
            outcome.evaluated_control_points
        );
        assert!(
            outcome.inlier_area_coverage > 0.0,
            "需求 2.8 coverage must be measurable once control points are accepted"
        );
        // 需求 2.7 keeps a local field that lowered the frame's median symmetric
        // reprojection error.  The baseline is the error of the global model, so a
        // frame the refinement actually fixed must not be reverted.
        assert!(
            !outcome.reverted_to_global,
            "a refinement that removed a real 2px displacement must be kept (global {:.2}px, local {:.2}px)",
            outcome.global_median_symmetric_error_px, outcome.local_median_symmetric_error_px
        );
        // The comparison has to be made against a baseline that exists.  The
        // regression this guards reverted every frame because the coarse pass
        // accepted too few control points, `median_of(&[])` returned `0.0`, and
        // no refinement can be below zero.
        assert!(
            outcome.global_baseline_samples >= intra_station::INTRA_STATION_MIN_BASELINE_SAMPLES,
            "需求 2.7 needs a real global baseline, not an empty median ({} samples)",
            outcome.global_baseline_samples
        );
        assert_eq!(
            outcome.local_field_verdict,
            intra_station::LocalFieldVerdict::Kept
        );
        assert_eq!(outcome.frame_status(), IntraStationFrameStatus::Local);
    }

    /// 需求 2.2 caps the *actual* intra-station analysis resolution at 2048
    /// while the production compositors currently configure 1400. This also
    /// guards Stack_Report's source value: both paths share the registrar's
    /// actual configuration instead of reporting the ceiling.
    #[test]
    fn the_analysis_budgets_stay_inside_the_intra_station_bound() {
        const {
            assert!(
                intra_station::INTRA_STATION_ANALYSIS_LONG_SIDE
                    <= intra_station::INTRA_STATION_ANALYSIS_MAX_LONG_SIDE
            );
            assert!(ANALYSIS_LONG_SIDE == intra_station::INTRA_STATION_ANALYSIS_LONG_SIDE);
            assert!(
                STREAMING_ANALYSIS_LONG_SIDE == intra_station::INTRA_STATION_ANALYSIS_LONG_SIDE
            );
            assert!(ANALYSIS_LONG_SIDE <= intra_station::INTRA_STATION_ANALYSIS_MAX_LONG_SIDE);
            assert!(
                STREAMING_ANALYSIS_LONG_SIDE <= intra_station::INTRA_STATION_ANALYSIS_MAX_LONG_SIDE
            );
        };
    }

    /// The tiled compositor is the *only* path a multi-station stack ever takes,
    /// so 需求 2.3–2.9 has to be measurable there.  Before this the native
    /// refinement existed solely on the in-memory path, and every real focus
    /// stack — three camera positions, an 8369×10114 canvas — skipped it
    /// silently and produced output byte identical to a run without it.
    #[test]
    fn streaming_intra_station_refinement_measures_across_tiles() {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        let width = 1_200u32;
        let height = 256u32;
        let base = refine_texture(width, height);
        let store = StreamingMosaicStore::new(width, height).expect("streaming store");
        assert!(
            store.tile_columns > 1,
            "the fixture must straddle a tile boundary"
        );
        for tile_row in 0..store.tile_rows {
            for tile_column in 0..store.tile_columns {
                let (left, top, tile_width, tile_height) = store.tile_extent(tile_column, tile_row);
                let tile = Rgb32FImage::from_fn(tile_width, tile_height, |x, y| {
                    *base.get_pixel(left + x, top + y)
                });
                let covered = GrayImage::from_pixel(tile_width, tile_height, Luma([255]));
                let owner = GrayImage::from_pixel(tile_width, tile_height, Luma([1]));
                store
                    .save_tile(tile_column, tile_row, &tile, &covered)
                    .expect("tile");
                store
                    .save_owner_tile(tile_column, tile_row, &owner)
                    .expect("owner tile");
            }
        }
        let shifted = Rgb32FImage::from_fn(width, height, |x, y| {
            *base.get_pixel((x + 2).min(width - 1), y)
        });
        let info = image_info(1, &shifted);
        fn layer_sampler<'a>(
            source: &'a Rgb32FImage,
            info: &'a ImageInfo,
            width: u32,
            height: u32,
        ) -> LayerSampler<'a> {
            LayerSampler {
                info,
                source,
                source_divisor: 1.0,
                inverse: Matrix3::identity(),
                native_mapping: None,
                projection: Projection::Planar,
                offset: (0.0, 0.0),
                left: 0,
                top: 0,
                scale: 1.0,
                residual: Field::new(width, height, GRID_STEP as f64),
            }
        }
        let mode = NativeRefineMode::IntraStation {
            anchor_long_side: 256,
        };
        let mut sampler = layer_sampler(&shifted, &info, width, height);
        let outcome = refine_native_layer_streaming(
            &store,
            Some(1),
            &mut sampler,
            width,
            height,
            (0, width - 1, 0, height - 1),
            mode,
        )
        .expect("streaming native refinement");
        assert!(
            outcome.accepted_control_points >= 16,
            "the tiled path must measure real control points ({}/{} accepted)",
            outcome.accepted_control_points,
            outcome.evaluated_control_points
        );
        assert!(!outcome.reverted_to_global);
        assert!(outcome.inlier_area_coverage > 0.20);
        // 需求 2.2: a later focal plane registers against its own camera
        // position, so canvas owned by another station is not a base at all.
        let mut foreign = layer_sampler(&shifted, &info, width, height);
        let ignored = refine_native_layer_streaming(
            &store,
            Some(7),
            &mut foreign,
            width,
            height,
            (0, width - 1, 0, height - 1),
            mode,
        )
        .expect("streaming native refinement");
        assert_eq!(ignored.accepted_control_points, 0);
        assert_eq!(ignored.evaluated_control_points, 0);
    }

    #[test]
    fn native_registration_observes_the_final_depth_band_inverse() {
        let source = Rgb32FImage::new(128, 128);
        let info = image_info(1, &source);
        let mapping = |target: Point3<f64>| Some(Point2::new(target.x + 7.0, target.y - 3.0));
        let mut residual = Field::new(64, 64, 16.0);
        residual.values.fill([1.0, 2.0]);
        let sampler = LayerSampler {
            info: &info,
            source: &source,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            native_mapping: Some(&mapping),
            projection: Projection::Planar,
            offset: (4.0, 5.0),
            left: 10,
            top: 20,
            scale: 2.0,
            residual,
        };
        // Canvas residual precedes the same depth map as final rendering:
        // (10+30+2-4, 20+40+4-5) then (+7,-3).
        assert_eq!(
            sampler.source_point(30.0, 40.0),
            Some(Point2::new(45.0, 56.0))
        );
        let global = LayerSampler {
            native_mapping: None,
            ..sampler
        };
        assert_eq!(
            global.source_point(30.0, 40.0),
            Some(Point2::new(38.0, 59.0))
        );
    }

    #[test]
    fn sensor_noise_cannot_hide_a_sharper_faint_colour_stroke() {
        let (sharp, noisy_blur) = noisy_colour_stroke_pair();
        let info = image_info(1, &sharp);
        let sampler = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            native_mapping: None,
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(192, 160, 48.0),
        };
        let mask = GrayImage::from_pixel(192, 160, Luma([255]));
        let chosen = ownership(
            &noisy_blur,
            &mask,
            &sampler,
            &Field::new(192, 160, 56.0),
            192,
            160,
        );
        let selected = (30..130)
            .filter(|&y| chosen.get_pixel(96, y)[0] > 0)
            .count();
        assert!(
            selected >= 90,
            "faint focused colour must beat noisy defocus ({selected}/100)"
        );
    }

    #[test]
    fn extra_sensor_noise_cannot_reclaim_an_already_focused_stroke() {
        let (sharp, noisy_blur) = noisy_colour_stroke_pair();
        let info = image_info(1, &noisy_blur);
        let sampler = LayerSampler {
            info: &info,
            source: &noisy_blur,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            native_mapping: None,
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(192, 160, 48.0),
        };
        let mask = GrayImage::from_pixel(192, 160, Luma([255]));
        let chosen = ownership(
            &sharp,
            &mask,
            &sampler,
            &Field::new(192, 160, 56.0),
            192,
            160,
        );
        let selected = (30..130)
            .filter(|&y| chosen.get_pixel(96, y)[0] > 0)
            .count();
        assert_eq!(
            selected, 0,
            "noisy defocus must not overwrite the focused stroke"
        );
    }

    #[test]
    fn diffuse_halo_cannot_outvote_a_narrow_focused_contour() {
        let sharp = Rgb32FImage::from_fn(192, 160, |x, y| {
            let line = (x as i32 - 96).abs() <= 1 && (20..140).contains(&y);
            let paper = 0.40 + 0.003 * (x as f32 * 0.17).sin();
            let value = if line { 0.28 } else { paper };
            Rgb([value, value * 0.9, value * 0.8])
        });
        let base = image::imageops::blur(&sharp, 5.0);
        let info = image_info(1, &sharp);
        let sampler = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            native_mapping: None,
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(192, 160, 48.0),
        };
        let chosen = ownership(
            &base,
            &GrayImage::from_pixel(192, 160, Luma([255])),
            &sampler,
            &Field::new(192, 160, 56.0),
            192,
            160,
        );
        let selected = (30..130)
            .filter(|&y| chosen.get_pixel(96, y)[0] > 0)
            .count();
        assert!(
            selected >= 90,
            "focused contour lost to its spread halo: {selected}/100"
        );
    }

    #[test]
    fn enclosed_sharp_tile_replaces_blur_without_losing_coverage() {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        // Four times the old 160x120 fixture, with the candidate a 320x320 crop
        // instead of an 80x80 one.  On the small canvas the native control
        // points had to sit at least one 57x57 patch away from the layer edge,
        // which left 2 measurable cells out of 18 — an 11% inlier spatial
        // support that is an artefact of an 80px layer, not of the photographs,
        // and that 需求 2.8 / 12.2 correctly refuse to fuse.  A 320px layer
        // measures 0.4 of its cells, so the test exercises the selection it is
        // about; the exclusion path has its own unit tests in
        // `stack_pipeline::station_degradation`.
        let sharp = Rgb32FImage::from_fn(640, 480, |x, y| {
            let value = 0.36
                + 0.12 * (x as f32 * 0.75).sin()
                + 0.10 * (y as f32 * 0.65).cos()
                + 0.03 * ((x + y) as f32 * 0.19).sin();
            Rgb([value * 1.1, value, value * 0.8])
        });
        let base = image::imageops::blur(&sharp, 2.0);
        let candidate = image::imageops::crop_imm(&sharp, 160, 80, 320, 320).to_image();
        let sources = [base.clone(), candidate];
        let infos = [image_info(0, &sources[0]), image_info(1, &sources[1])];
        let transforms = HashMap::from([
            (0, Matrix3::identity()),
            (
                1,
                Matrix3::new(1.0, 0.0, 160.0, 0.0, 1.0, 80.0, 0.0, 0.0, 1.0),
            ),
        ]);
        let app = tauri::test::mock_app();
        let output = detail_preserving_mosaic(
            &[&infos[0], &infos[1]],
            &transforms,
            Projection::Planar,
            None,
            false,
            app.handle().clone(),
            "test-progress",
            &mut |info| Ok(sources[info.id].clone()),
        )
        .unwrap();
        assert_eq!(
            output.dimensions(),
            sharp.dimensions(),
            "coverage must include the last row and column"
        );
        let mut before = 0.0;
        let mut after = 0.0;
        for y in 120..360 {
            for x in 200..440 {
                before += (base.get_pixel(x, y)[1] - sharp.get_pixel(x, y)[1]).powi(2);
                after += (output.get_pixel(x, y)[1] - sharp.get_pixel(x, y)[1]).powi(2);
            }
        }
        assert!(
            after < before * 0.05,
            "the sharp interior must replace the blur: {after} vs {before}"
        );
        assert!(
            output
                .pixels()
                .all(|p| p.0.iter().all(|v| v.is_finite() && *v > 0.0))
        );
    }

    #[test]
    fn blurred_candidate_cannot_replace_an_already_sharp_interior() {
        let base = sharp_fixture();
        let source = image::imageops::blur(&base, 2.0);
        let info = image_info(0, &source);
        let sampler = LayerSampler {
            info: &info,
            source: &source,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            native_mapping: None,
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(160, 120, 48.0),
        };
        let mask = GrayImage::from_pixel(160, 120, Luma([255]));
        let selected = ownership(
            &base,
            &mask,
            &sampler,
            &Field::new(160, 120, 56.0),
            160,
            120,
        );
        let switched = (10..110)
            .flat_map(|y| (10..150).map(move |x| (x, y)))
            .filter(|&(x, y)| selected.get_pixel(x, y)[0] != 0)
            .count();
        assert_eq!(switched, 0);
    }

    #[test]
    fn sparse_in_focus_stroke_is_not_lost_to_cell_background() {
        let sharp = Rgb32FImage::from_fn(160, 120, |x, y| {
            let line = (x as i32 - 80).unsigned_abs() <= 1 && (12..108).contains(&y);
            let value = if line { 0.04 } else { 0.42 };
            Rgb([value, value, value])
        });
        let base = image::imageops::blur(&sharp, 3.0);
        let info = image_info(1, &sharp);
        let sampler = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            native_mapping: None,
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(160, 120, 48.0),
        };
        let mask = GrayImage::from_pixel(160, 120, Luma([255]));
        let selected = ownership(
            &base,
            &mask,
            &sampler,
            &Field::new(160, 120, 56.0),
            160,
            120,
        );
        let selected_on_stroke = (12..108)
            .filter(|&y| selected.get_pixel(80, y)[0] != 0)
            .count();
        assert!(
            selected_on_stroke > 70,
            "a sparse sharp stroke must own its pixels ({selected_on_stroke}/96)"
        );
    }

    #[test]
    fn streaming_focus_probe_keeps_a_sparse_sharp_stroke() {
        let sharp = Rgb32FImage::from_fn(160, 128, |x, y| {
            let line = (x as i32 - 80).unsigned_abs() <= 1 && (8..120).contains(&y);
            let value = if line { 0.04 } else { 0.42 };
            Rgb([value, value, value])
        });
        let base = image::imageops::blur(&sharp, 3.0);
        let info = image_info(1, &sharp);
        let sampler = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            native_mapping: None,
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(160, 128, 48.0),
        };
        let selected = streaming_ownership_with_long_side(
            &base,
            &GrayImage::from_pixel(160, 128, Luma([255])),
            &sampler,
            &Field::new(160, 128, 56.0),
            160,
            128,
            5,
        );
        assert!(
            selected.get_pixel(2, 1)[0] != 0 || selected.get_pixel(2, 2)[0] != 0,
            "the bounded streaming probe must retain sparse focused writing"
        );
    }

    #[test]
    fn disagreement_is_lower_for_aligned_defocus_than_for_shifted_detail() {
        let sharp = Rgb32FImage::from_fn(160, 120, |x, y| {
            let value = 0.35 + 0.22 * (x as f32 * 0.37).sin() + 0.16 * (y as f32 * 0.29).cos();
            Rgb([value, value * 0.9, value * 0.75])
        });
        let base = image::imageops::blur(&sharp, 2.5);
        let info = image_info(1, &sharp);
        let tone = Field::new(160, 120, 56.0);
        let aligned = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            native_mapping: None,
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(160, 120, 48.0),
        };
        let aligned_error = ownership_disagreement(&base, &aligned, &tone, 32.0, 32.0, 40.0);
        let shifted = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::new(1.0, 0.0, 4.0, 0.0, 1.0, -3.0, 0.0, 0.0, 1.0),
            native_mapping: None,
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(160, 120, 48.0),
        };
        let shifted_error = ownership_disagreement(&base, &shifted, &tone, 32.0, 32.0, 40.0);
        assert!(
            aligned_error < shifted_error,
            "local averaging should preserve geometric mismatch signal: {aligned_error} vs {shifted_error}"
        );
    }

    #[test]
    fn invalid_sample_gap_cannot_cancel_disconnected_colour_mismatches() {
        let base = Rgb32FImage::from_pixel(100, 100, Rgb([0.5; 3]));
        let source = Rgb32FImage::from_fn(100, 100, |x, _| {
            // Invalid samples must retain their positions. If the valid
            // samples are packed into a smaller grid, these separated light
            // and dark strips become neighbours and falsely cancel out.
            let value = if x < 30 {
                0.8
            } else if x > 60 {
                0.2
            } else {
                f32::NAN
            };
            Rgb([value; 3])
        });
        let info = image_info(1, &source);
        let sampler = LayerSampler {
            info: &info,
            source: &source,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            native_mapping: None,
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(100, 100, 48.0),
        };
        let error =
            ownership_disagreement(&base, &sampler, &Field::new(100, 100, 56.0), 0.0, 0.0, 80.0);
        assert!(
            (error - 0.3).abs() < 1e-6,
            "missing samples must not hide a real colour mismatch: {error}"
        );
    }

    #[test]
    fn fully_invalid_overlap_is_not_treated_as_an_aligned_match() {
        let source = Rgb32FImage::from_pixel(32, 32, Rgb([0.5; 3]));
        let info = image_info(1, &source);
        let sampler = LayerSampler {
            info: &info,
            source: &source,
            source_divisor: 1.0,
            inverse: Matrix3::new(1.0, 0.0, 64.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            native_mapping: None,
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(32, 32, 48.0),
        };
        assert_eq!(
            ownership_disagreement(&source, &sampler, &Field::new(32, 32, 56.0), 0.0, 0.0, 24.0,),
            1.0,
            "absence of common samples must not remove the mismatch penalty"
        );
    }

    #[test]
    fn scaled_sparse_detail_is_sampled_at_the_native_cell_centre() {
        let sharp = Rgb32FImage::from_fn(3_200, 1_600, |x, _| {
            let value = if (x as i32 - 84).unsigned_abs() <= 2 {
                0.04
            } else {
                0.42
            };
            Rgb([value; 3])
        });
        let base = image::imageops::blur(&sharp, 4.0);
        let info = image_info(1, &sharp);
        let sampler = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            native_mapping: None,
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 8.0,
            residual: Field::new(400, 200, 48.0),
        };
        let selected = ownership(
            &base,
            &GrayImage::from_pixel(3_200, 1_600, Luma([255])),
            &sampler,
            &Field::new(400, 200, 56.0),
            400,
            200,
        );
        let selected_on_stroke = (20..180)
            .filter(|&y| selected.get_pixel(10, y)[0] != 0)
            .count();
        assert!(
            selected_on_stroke > 100,
            "a native sharp stroke at a scaled cell centre must own its detail ({selected_on_stroke}/160)"
        );
    }

    #[test]
    fn smooth_tone_correction_preserves_selected_detail_relative_contrast() {
        let p = Rgb([0.2, 0.3, 0.4]);
        let q = Rgb([0.4, 0.5, 0.6]);
        let a = adjusted(p, [0.1, -0.05, 0.02]);
        let b = adjusted(q, [0.1, -0.05, 0.02]);
        for c in 0..3 {
            assert!((b[c] / a[c] - q[c] / p[c]).abs() < 1e-6);
        }
    }

    #[test]
    fn displacement_field_interpolation_is_continuous_at_cell_edges() {
        let mut field = Field::<2>::new(200, 200, 48.0);
        for (i, value) in field.values.iter_mut().enumerate() {
            *value = [
                (i % field.width) as f64 * 0.3,
                (i / field.width) as f64 * -0.2,
            ];
        }
        let a = field.at(48.0 - 1e-5, 73.0);
        let b = field.at(48.0 + 1e-5, 73.0);
        assert!((a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6);
    }
}
