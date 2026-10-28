//! Pure Quality_Gate measurements over already-decoded pixels and geometry.
//!
//! This module does not decode sources, read settings, mutate ownership, write
//! reports, or decide whether an export proceeds. Callers supply display-sRGB
//! float pixels, native/world geometry, coverage and immutable owner planes.
//! All error identifiers reuse the existing report/degradation vocabulary.
#![allow(dead_code)] // Production wiring belongs to tasks 15.7/15.8.

pub(crate) const QUALITY_LOCAL_SCALE_MEDIAN_MIN: f64 = 0.98;
pub(crate) const QUALITY_LOCAL_SCALE_PIXEL_MIN: f64 = 0.95;
pub(crate) const QUALITY_LOCAL_SCALE_PIXEL_RATIO_MIN: f64 = 0.99;
pub(crate) const QUALITY_EFFECTIVE_PIXEL_RATIO_MIN: f64 = 0.98;
pub(crate) const QUALITY_MTF50_RATIO_MIN: f64 = 0.93;
pub(crate) const QUALITY_GRADIENT_RATIO_MIN: f64 = 0.95;
pub(crate) const QUALITY_NOISE_RATIO_MIN: f64 = 0.85;
pub(crate) const QUALITY_NOISE_RATIO_MAX: f64 = 1.15;
pub(crate) const QUALITY_ROI_DELTA_E00_MAX: f64 = 2.0;
pub(crate) const QUALITY_BOUNDARY_P95_MAX: f64 = 1.5;
pub(crate) const QUALITY_BOUNDARY_MAX: f64 = 3.0;
/// 需求 11.12: at most 1% of Textured_Pixel may have an owner more than 5%
/// below the cell's best candidate.
pub(crate) const QUALITY_OWNER_SHORTFALL_RATIO_MAX: f64 = 0.01;
pub(crate) const OWNER_SHARPNESS_SHORTFALL_MAX: f32 = 0.05;
pub(crate) const OWNER_SHARPNESS_COVERAGE_MIN: f64 = 1.0 - QUALITY_OWNER_SHORTFALL_RATIO_MAX;
pub(crate) const QUALITY_MIN_MEASURABLE: usize = 8;
pub(crate) const QUALITY_MAX_UNMEASURABLE_RATIO: f64 = 0.20;
pub(crate) const QUALITY_CRITERIA: [&str; 9] = [
    "local_scale_median",
    "local_scale_pixel_ratio",
    "effective_pixel_count",
    "mtf50_normalized",
    "gradient_energy_normalized",
    "noise_sigma_ratio",
    "roi_delta_e00",
    "boundary_stroke_alignment",
    "owner_sharpness_coverage",
];

/// Diagnostic measurements remain visible but cannot establish acceptance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RatioMeasurement {
    pub value: f64,
    pub passed: bool,
    pub diagnostic: bool,
}

/// Ratios require a strictly positive, measurable reference. Zero/zero is
/// not evidence of a passing MTF, gradient or noise comparison.
pub(crate) fn compare_ratio(
    measured: f64,
    reference: f64,
    minimum: f64,
    maximum: f64,
    diagnostic: bool,
) -> Option<RatioMeasurement> {
    if !measured.is_finite() || measured < 0.0 || !reference.is_finite() || reference <= 0.0 {
        return None;
    }
    let value = measured / reference;
    value.is_finite().then_some(RatioMeasurement {
        value,
        passed: !diagnostic && value >= minimum && value <= maximum,
        diagnostic,
    })
}

#[cfg(test)]
use super::degradation;
#[cfg(test)]
use image::{GrayImage, Rgb32FImage};

mod roi_geometry;

pub(crate) use roi_geometry::{
    EffectivePixelCoverage, LocalScaleMeasurement, OwnerRegions, QUALITY_ROI_MARGIN,
    QUALITY_ROI_MAX_COUNT, QUALITY_ROI_SIDE, QualityRoi, SourceGeometry, label_owner_regions,
    local_scale_with_mapper, map_world_to_source, map_world_to_source_with_inverse,
    select_quality_rois,
};
#[cfg(test)]
pub(crate) use roi_geometry::{
    effective_pixel_coverage, local_scale_for_roi, map_roi_corners_to_source,
};

mod pixel_metrics;

#[cfg(test)]
use pixel_metrics::metric_luminance;
#[cfg(test)]
pub(crate) use pixel_metrics::{FLAT_LOW_FREQUENCY_STD_MAX, detect_slanted_edge, noise_sigma};
pub(crate) use pixel_metrics::{
    flat_roi, noise_sigma_ratio, normalized_gradient_energy, roi_low_frequency_delta_e00,
    roi_mean_rgb, slanted_edge_mtf50,
};

mod boundary_strokes;

#[cfg(test)]
use boundary_strokes::paired_boundary_edge_error;
#[cfg(test)]
pub(crate) use boundary_strokes::{
    measure_boundary_strokes, paired_boundary_edge_fits_with_coverage,
};
pub(crate) use boundary_strokes::{
    measure_boundary_strokes_with_coverage, paired_boundary_edge_error_with_coverage,
    trace_moore_boundary,
};

#[cfg(test)]
mod gradient_tests;

#[cfg(test)]
mod boundary_edge_tests;
