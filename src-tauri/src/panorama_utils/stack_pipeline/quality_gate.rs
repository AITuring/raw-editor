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
mod gradient_tests {
    use super::*;
    use image::{Rgb, Rgb32FImage};

    #[test]
    fn cached_gradient_matches_the_production_acutance_primitive() {
        let mut image = Rgb32FImage::new(17, 19);
        for (i, pixel) in image.pixels_mut().enumerate() {
            let value = ((i * 37 % 251) as f32) / 251.0;
            *pixel = Rgb([value, value * 0.93, value * 0.87]);
        }
        let plane = metric_luminance(&image).unwrap();
        let width = image.width() as usize;
        let height = image.height() as usize;
        let expected = (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .map(|(x, y)| {
                crate::panorama_utils::mosaic::acutance_with_step(
                    |sx, sy| {
                        let nx = (sx.round() as isize).clamp(0, width as isize - 1) as usize;
                        let ny = (sy.round() as isize).clamp(0, height as isize - 1) as usize;
                        Some(Rgb([plane[ny * width + nx] as f32; 3]))
                    },
                    x as f64,
                    y as f64,
                    1.0,
                )
            })
            .sum::<f64>()
            / plane.len() as f64;
        let actual = normalized_gradient_energy(&image, 1.0).unwrap();
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }
}

#[cfg(test)]
mod boundary_edge_tests {
    use super::*;
    fn edge_image(shift: f64) -> Rgb32FImage {
        Rgb32FImage::from_fn(128, 128, |x, y| {
            let amplitude = 0.5 + 0.4 * ((x as f64 - 56.0) / 4.0).tanh();
            let position = 64.35 + if x >= 64 { shift } else { 0.0 };
            let value = 0.05 + amplitude * (0.5 + 0.5 * ((y as f64 - position) * 2.0).tanh());
            image::Rgb([value as f32; 3])
        })
    }
    #[test]
    fn continuous_edge_keeps_its_position_when_contrast_changes() {
        let image = edge_image(0.0);
        let (error, _) = paired_boundary_edge_error(&image, 64, 64, (1.0, 0.0)).unwrap();
        assert!(
            error < 0.35,
            "illumination was mistaken for edge displacement: {error}"
        );
    }
    #[test]
    fn common_contour_is_not_replaced_by_a_different_stronger_parallel_edge() {
        let image = Rgb32FImage::from_fn(128, 128, |x, y| {
            let (first, second) = if x < 64 { (0.4, 0.5) } else { (0.5, 0.4) };
            let step = |position: f64| 0.5 + 0.5 * ((y as f64 - position) * 2.0).tanh();
            image::Rgb([(0.05 + first * step(51.35) + second * step(64.35)) as f32; 3])
        });
        let (error, _) = paired_boundary_edge_error(&image, 64, 64, (1.0, 0.0)).unwrap();
        assert!(
            error < 0.35,
            "different continuous contours were paired: {error}"
        );
    }

    #[test]
    fn isolated_fifteen_pixel_edge_jump_remains_a_measurable_failure() {
        let (error, _) = paired_boundary_edge_error(&edge_image(15.0), 64, 64, (1.0, 0.0)).unwrap();
        // This edge sits one pixel from the 16px band's end, so its
        // truncated derivative footprint can bias the subpixel fit.
        assert!(
            (14.0..16.0).contains(&error),
            "large jump was hidden: {error}"
        );
        assert!(error > QUALITY_BOUNDARY_MAX);
    }
    #[test]
    fn one_unrelated_bright_pixel_cannot_hide_an_isolated_edge_jump() {
        let mut image = edge_image(15.0);
        image.put_pixel(78, 64, image::Rgb([1.0; 3]));
        let (error, _) = paired_boundary_edge_error(&image, 64, 64, (1.0, 0.0)).unwrap();
        assert!(
            error > QUALITY_BOUNDARY_MAX,
            "an unrelated glint hid the jump: {error}"
        );
    }
    #[test]
    fn covered_interpolation_halo_does_not_create_a_photographic_stroke() {
        let image = Rgb32FImage::from_fn(128, 128, |_, y| {
            image::Rgb([if y < 62 { 0.0 } else { 0.9 }; 3])
        });
        let halo = GrayImage::from_fn(128, 128, |_, y| image::Luma([if y < 60 { 0 } else { 255 }]));
        assert!(
            paired_boundary_edge_error_with_coverage(&image, Some(&halo), 64, 64, (1.0, 0.0))
                .is_err()
        );
        let full_support =
            GrayImage::from_fn(128, 128, |_, y| image::Luma([if y < 40 { 0 } else { 255 }]));
        assert!(
            paired_boundary_edge_error_with_coverage(
                &image,
                Some(&full_support),
                64,
                64,
                (1.0, 0.0)
            )
            .unwrap()
            .0 < 1e-6
        );
    }

    #[test]
    fn transparency_edges_are_not_strokes_but_covered_black_edges_are() {
        let image = Rgb32FImage::from_fn(128, 128, |_, y| {
            image::Rgb([if y < 64 { 0.0 } else { 0.9 }; 3])
        });
        let coverage =
            GrayImage::from_fn(128, 128, |_, y| image::Luma([if y < 64 { 0 } else { 255 }]));
        assert!(
            paired_boundary_edge_error_with_coverage(&image, Some(&coverage), 64, 64, (1.0, 0.0))
                .is_err()
        );
        let covered = GrayImage::from_pixel(128, 128, image::Luma([255]));
        let (error, _) =
            paired_boundary_edge_error_with_coverage(&image, Some(&covered), 64, 64, (1.0, 0.0))
                .unwrap();
        assert!(error < 1e-6);
    }

    #[test]
    fn parallel_segments_outside_the_seam_window_are_not_a_crossing_pair() {
        let image = Rgb32FImage::from_fn(128, 128, |x, y| {
            let d = y as f64 - 64.0;
            let edge = 64.0
                + if y < 64 {
                    25.0 + 2.0 * d
                } else {
                    -25.0 + 2.0 * d
                };
            image::Rgb([if x as f64 >= edge { 0.9 } else { 0.1 }; 3])
        });
        assert_eq!(
            paired_boundary_edge_error(&image, 64, 64, (0.0, 1.0)),
            Err(degradation::BOUNDARY_NO_PAIRABLE_EDGE)
        );
    }

    #[test]
    fn different_parallel_edges_are_not_one_crossing_stroke() {
        let image = Rgb32FImage::from_fn(128, 128, |_, y| {
            image::Rgb([if y <= 20 || y >= 44 { 0.9 } else { 0.1 }; 3])
        });
        assert!(paired_boundary_edge_error(&image, 64, 32, (0.0, 1.0)).is_err());
    }

    #[test]
    fn shifted_edge_is_still_rejected_at_the_original_threshold() {
        let image = edge_image(3.0);
        let (error, _) = paired_boundary_edge_error(&image, 64, 64, (1.0, 0.0)).unwrap();
        assert!(
            (error - 3.0).abs() < 0.35,
            "wrong native edge displacement: {error}"
        );
        assert!(error > QUALITY_BOUNDARY_P95_MAX);
    }
}
