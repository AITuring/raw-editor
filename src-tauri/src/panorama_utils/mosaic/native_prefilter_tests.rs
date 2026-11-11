//! Native coarse prefilter geometry, support and measured correspondence.

use super::*;
use crate::panorama_utils::mosaic::Field;
use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::stitching::{Projection, get_high_quality_interpolated_pixel};
use image::{Rgb, Rgb32FImage};

fn info(image: &Rgb32FImage) -> ImageInfo {
    ImageInfo {
        id: 0,
        filename: "prefilter-fixture.png".into(),
        width: image.width(),
        height: image.height(),
        alignment_image: GrayImage::new(1, 1),
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

fn sampler<'a>(image: &'a Rgb32FImage, info: &'a ImageInfo) -> LayerSampler<'a> {
    LayerSampler {
        info,
        source: image,
        source_divisor: 1.0,
        inverse: Matrix3::identity(),
        native_mapping: None,
        projection: Projection::Planar,
        offset: (0.0, 0.0),
        left: 0,
        top: 0,
        scale: 1.0,
        residual: Field::new(image.width(), image.height(), 32.0),
    }
}

fn window<'a>(image: &'a Rgb32FImage, mask: &'a GrayImage) -> NativeRefineBase<'a> {
    NativeRefineBase {
        image,
        mask,
        left: 0,
        top: 0,
    }
}

#[test]
fn native_prefilter_preserves_constant_ramp_centres_and_pair_symmetry() {
    let a = Rgb32FImage::from_fn(64, 64, |x, y| {
        Rgb([0.2 + 0.003 * x as f32 + 0.002 * y as f32; 3])
    });
    let b = Rgb32FImage::from_pixel(64, 64, Rgb([0.4; 3]));
    let mask = GrayImage::from_pixel(64, 64, Luma([255]));
    let (ai, bi) = (info(&a), info(&b));
    for (x, y) in [(16.0, 17.0), (16.25, 17.75), (16.5, 17.5)] {
        let pair =
            native_patch_sample_pair(&window(&a, &mask), &sampler(&b, &bi), x, y, true).unwrap();
        let reversed =
            native_patch_sample_pair(&window(&b, &mask), &sampler(&a, &ai), x, y, true).unwrap();
        assert!((pair.0 - (0.2 + 0.003 * x + 0.002 * y)).abs() < 1e-7);
        assert!((pair.1 - 0.4).abs() < 1e-7);
        assert_eq!(pair.0.to_bits(), reversed.1.to_bits());
        assert_eq!(pair.1.to_bits(), reversed.0.to_bits());
    }
}

#[test]
fn native_prefilter_checks_the_whole_cubic_support_without_partial_renormalisation() {
    let image = Rgb32FImage::from_pixel(32, 32, Rgb([0.4; 3]));
    let image_info = info(&image);
    let source = sampler(&image, &image_info);
    let mut mask = GrayImage::from_pixel(32, 32, Luma([255]));
    // This pixel is neither the centre nor one of the four tap floors. It is
    // nevertheless read by the cubic interpolation at (9.5, 9.5).
    mask.put_pixel(8, 8, Luma([0]));
    let base = window(&image, &mask);
    assert!(
        base.covered(10, 10),
        "the original coverage denominator remains intact"
    );
    assert!(native_patch_sample_pair(&base, &source, 10.0, 10.0, false).is_some());
    assert!(native_patch_sample_pair(&base, &source, 10.0, 10.0, true).is_none());
    let full_mask = GrayImage::from_pixel(32, 32, Luma([255]));
    let base = window(&image, &full_mask);
    assert!(native_patch_sample_pair(&base, &source, 0.0, 10.0, true).is_none());
    assert!(native_patch_sample_pair(&base, &source, 31.0, 10.0, true).is_none());

    // All tap centres lie inside the declared source, but the cubic neighbour
    // at x=15 does not. Source support must be checked as well as the base mask.
    let mut clipped_info = info(&image);
    clipped_info.width = 15;
    let clipped = sampler(&image, &clipped_info);
    assert!(native_patch_sample_pair(&base, &clipped, 13.0, 10.0, false).is_some());
    assert!(native_patch_sample_pair(&base, &clipped, 13.0, 10.0, true).is_none());
}

#[test]
fn native_prefilter_leaves_finest_and_legacy_samples_bit_identical() {
    let image = Rgb32FImage::from_fn(32, 32, |x, y| {
        Rgb([0.3 + 0.1 * (x as f32 * 0.7 + y as f32 * 0.4).sin(); 3])
    });
    let image_info = info(&image);
    let source = sampler(&image, &image_info);
    let mask = GrayImage::from_pixel(32, 32, Luma([255]));
    let base = window(&image, &mask);
    let native = NativeRefineMode::IntraStation {
        anchor_long_side: 9_504,
    };
    assert!(uses_native_area_prefilter(native, 2.0));
    for (mode, spacing) in [
        (native, 1.0),
        (NativeRefineMode::LegacyMosaic, 1.0),
        (NativeRefineMode::LegacyMosaic, 2.0),
    ] {
        assert!(!uses_native_area_prefilter(mode, spacing));
        for (x, y) in [(8.0, 9.0), (8.25, 9.75), (8.5, 9.5)] {
            let pair = native_patch_sample_pair(
                &base,
                &source,
                x,
                y,
                uses_native_area_prefilter(mode, spacing),
            )
            .unwrap();
            assert_eq!(pair.0.to_bits(), luma(base.masked(x, y).unwrap()).to_bits());
            assert_eq!(
                pair.1.to_bits(),
                luma(source.sample(x, y).unwrap()).to_bits()
            );
        }
    }
}

#[test]
fn native_prefilter_reduces_weave_before_decimation() {
    let image = Rgb32FImage::from_fn(64, 64, |x, _| {
        Rgb([0.5 + 0.2 * (std::f64::consts::TAU * 0.45 * f64::from(x)).cos() as f32; 3])
    });
    let image_info = info(&image);
    let source = sampler(&image, &image_info);
    let mask = GrayImage::from_pixel(64, 64, Luma([255]));
    let base = window(&image, &mask);
    let mut point_energy = 0.0;
    let mut area_energy = 0.0;
    for x in (12..52).step_by(2) {
        let point = native_patch_sample_pair(&base, &source, f64::from(x), 32.0, false).unwrap();
        let area = native_patch_sample_pair(&base, &source, f64::from(x), 32.0, true).unwrap();
        point_energy += (point.0 - 0.5).powi(2);
        area_energy += (area.0 - 0.5).powi(2);
    }
    assert!(
        area_energy < point_energy * 0.1,
        "{area_energy} vs {point_energy}"
    );
}

fn grain(seed: u32) -> Rgb32FImage {
    let image = Rgb32FImage::from_fn(256, 256, |x, y| {
        let hash = x.wrapping_mul(0x9e37_79b9) ^ y.wrapping_mul(0x85eb_ca6b) ^ seed;
        let hash = (hash ^ (hash >> 15)).wrapping_mul(0x7feb_352d);
        Rgb([0.2 + 0.6 * ((hash >> 8) & 0xffff) as f32 / 65_535.0; 3])
    });
    image::imageops::blur(&image, 1.2)
}

fn measured(base: &Rgb32FImage, layer: &Rgb32FImage) -> NativeControlPoint {
    let source_info = info(layer);
    let source = sampler(layer, &source_info);
    let mask = GrayImage::from_pixel(base.width(), base.height(), Luma([255]));
    measure_native_control_point(
        &window(base, &mask),
        &source,
        128,
        128,
        2.0,
        NativeRefineMode::IntraStation {
            anchor_long_side: 9_504,
        },
        intra_station::INTRA_STATION_MATCH_RADIUS,
        intra_station::INTRA_STATION_SEARCH_SAMPLES,
    )
}

#[test]
fn native_prefilter_measures_signed_subpixel_shifts_with_weave_and_defocus() {
    let mut base = grain(17);
    for (x, y, pixel) in base.enumerate_pixels_mut() {
        let weave = (0.025
            * (std::f64::consts::TAU * (0.43 * f64::from(x) + 0.11 * f64::from(y))).sin())
            as f32;
        for channel in &mut pixel.0 {
            *channel += weave;
        }
    }
    for (dx, dy) in [
        (0.25, 0.5),
        (-0.25, -0.5),
        (0.75, -0.25),
        (-0.75, 0.25),
        (1.25, -1.25),
        (-1.25, 1.25),
        (3.25, -3.25),
        (-3.25, 3.25),
    ] {
        let shifted = Rgb32FImage::from_fn(256, 256, |x, y| {
            get_high_quality_interpolated_pixel(&base, f64::from(x) - dx, f64::from(y) - dy)
        });
        let shifted = image::imageops::blur(&shifted, 0.65);
        let point = measured(&base, &shifted);
        let (delta, _, round_trip, _) = point
            .accepted
            .unwrap_or_else(|| panic!("shift=({dx},{dy}): {point:?}"));
        assert!((delta[0] - dx).abs() < 0.25, "dx={dx}, measured={delta:?}");
        assert!((delta[1] - dy).abs() < 0.25, "dy={dy}, measured={delta:?}");
        assert!(intra_station::bidirectional_consistent(round_trip));
    }
}

#[test]
fn native_prefilter_preserves_periodic_and_unrelated_rejections() {
    let base = grain(17);
    let unrelated = grain(91);
    assert!(measured(&base, &unrelated).accepted.is_none());
    let wave = |shift: f64| {
        Rgb32FImage::from_fn(256, 256, |x, y| {
            let phase = std::f64::consts::TAU / 8.0;
            Rgb([(0.5
                + 0.15 * ((f64::from(x) - shift) * phase).sin()
                + 0.15 * (f64::from(y) * phase).cos()) as f32; 3])
        })
    };
    let periodic = measured(&wave(0.0), &wave(1.25));
    assert!(periodic.accepted.is_none(), "{periodic:?}");
    assert!(
        periodic
            .peak_margin
            .is_some_and(|margin| margin < intra_station::INTRA_STATION_MIN_PEAK_MARGIN),
        "{periodic:?}"
    );
}

#[path = "native_prefilter_oracle_tests.rs"]
mod reuse_oracle;
