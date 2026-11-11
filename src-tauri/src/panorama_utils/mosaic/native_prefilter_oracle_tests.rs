//! Frozen scalar v20 sampling oracle. Keep the duplicate mapping and repeated
//! support checks here: they are the independent reference for their removal,
//! not a second production path.

use super::*;
use crate::panorama_utils::mosaic::{rgb_at, sampling};
use nalgebra::Point3;
use std::sync::Mutex;

fn scalar_sample(sampler: &LayerSampler<'_>, x: f64, y: f64) -> Option<Rgb<f32>> {
    let source = sampler.source_point(x, y)?;
    let sx = ((source.x + 0.5) / sampler.source_divisor - 0.5)
        .clamp(0.0, sampler.source.width() as f64 - 1.0);
    let sy = ((source.y + 0.5) / sampler.source_divisor - 0.5)
        .clamp(0.0, sampler.source.height() as f64 - 1.0);
    let pixel = rgb_at(sampler.source, sx, sy)?;
    sampling::corrected_lower_left_capture_sample(
        sampler.info,
        sampler.source,
        source.x,
        source.y,
        sx,
        sy,
        pixel,
    )
}

fn scalar_base_support(base: &NativeRefineBase<'_>, gx: f64, gy: f64) -> Option<()> {
    let (left, right, top, bottom) = interpolation_support(
        base.image.width(),
        base.image.height(),
        gx - f64::from(base.left),
        gy - f64::from(base.top),
    )?;
    for y in top..=bottom {
        for x in left..=right {
            if base.mask.get_pixel(x, y)[0] == 0
                || !base
                    .image
                    .get_pixel(x, y)
                    .0
                    .iter()
                    .all(|value| value.is_finite())
            {
                return None;
            }
        }
    }
    Some(())
}

fn scalar_source_support(sampler: &LayerSampler<'_>, x: f64, y: f64) -> Option<()> {
    let source = sampler.source_point(x, y)?;
    let divisor = sampler.source_divisor;
    let sx = (source.x + 0.5) / divisor - 0.5;
    let sy = (source.y + 0.5) / divisor - 0.5;
    let (left, right, top, bottom) =
        interpolation_support(sampler.source.width(), sampler.source.height(), sx, sy)?;
    for py in top..=bottom {
        for px in left..=right {
            let native = Point2::new(
                (f64::from(px) + 0.5) * divisor - 0.5,
                (f64::from(py) + 0.5) * divisor - 0.5,
            );
            let pixel = *sampler.source.get_pixel(px, py);
            if !sampler.info.source_is_supported(native)
                || !pixel.0.iter().all(|value| value.is_finite())
                || sampling::is_lower_left_black_capture_border(
                    sampler.source,
                    f64::from(px),
                    f64::from(py),
                    pixel,
                )
            {
                return None;
            }
        }
    }
    Some(())
}

fn scalar_pair(
    base: &NativeRefineBase<'_>,
    sampler: &LayerSampler<'_>,
    x: f64,
    y: f64,
    area_prefilter: bool,
) -> Option<(f64, f64)> {
    let gx = f64::from(sampler.left) + x;
    let gy = f64::from(sampler.top) + y;
    if !area_prefilter {
        return Some((
            luma(base.masked(gx, gy)?),
            luma(scalar_sample(sampler, x, y)?),
        ));
    }
    let mut pair = (0.0, 0.0);
    for oy in [-0.5, 0.5] {
        for ox in [-0.5, 0.5] {
            scalar_base_support(base, gx + ox, gy + oy)?;
            scalar_source_support(sampler, x + ox, y + oy)?;
            pair.0 += luma(base.masked(gx + ox, gy + oy)?) * 0.25;
            pair.1 += luma(scalar_sample(sampler, x + ox, y + oy)?) * 0.25;
        }
    }
    Some(pair)
}

fn pair_bits(pair: Option<(f64, f64)>) -> Option<(u64, u64)> {
    pair.map(|(a, b)| (a.to_bits(), b.to_bits()))
}

#[test]
fn reused_prefilter_is_bit_identical_with_origins_pyramids_and_nonidentity_maps() {
    let image = Rgb32FImage::from_fn(64, 64, |x, y| {
        Rgb([0.4 + 0.2 * (x as f32 * 0.7 + y as f32 * 0.2).sin(); 3])
    });
    let mut mask = GrayImage::from_pixel(64, 64, Luma([255]));
    mask.put_pixel(18, 18, Luma([0]));
    let base = NativeRefineBase {
        image: &image,
        mask: &mask,
        left: 19,
        top: 23,
    };
    let mapping = |p: Point3<f64>| {
        Some(Point2::new(
            2.0 * (p.x - 16.0) + 0.25 + (p.y * 0.03).sin(),
            2.0 * (p.y - 16.0) - 0.125,
        ))
    };
    let maps: [Option<&(dyn Fn(Point3<f64>) -> Option<Point2<f64>> + Sync)>; 2] =
        [None, Some(&mapping)];
    let mut successes = 0;
    let mut missing = 0;
    for divisor in [1.0, 2.0] {
        let mut image_info = info(&image);
        image_info.width = (64.0 * divisor) as u32;
        image_info.height = (64.0 * divisor) as u32;
        let mut source = sampler(&image, &image_info);
        source.source_divisor = divisor;
        source.left = 19;
        source.top = 23;
        source.offset = (3.0, 7.0);
        source.scale = 1.75;
        source.inverse = Matrix3::new(1.01, 0.02, -10.25, -0.01, 0.99, -9.75, 0.0001, -0.0002, 1.0);
        for (index, delta) in source.residual.values.iter_mut().enumerate() {
            *delta = [index as f64 * 0.015625, -(index as f64) * 0.03125];
        }
        for native_mapping in maps {
            source.native_mapping = native_mapping;
            for x in [-0.5, 0.0, 1.0, 10.25, 16.5, 18.0, 30.75, 61.5, 63.0] {
                for y in [0.25, 1.5, 10.0, 18.0, 30.75, 62.0] {
                    for filtered in [false, true] {
                        let expected = scalar_pair(&base, &source, x, y, filtered);
                        let actual = native_patch_sample_pair(&base, &source, x, y, filtered);
                        assert_eq!(
                            pair_bits(actual),
                            pair_bits(expected),
                            "divisor={divisor}, ({x},{y}), filtered={filtered}"
                        );
                        successes += usize::from(actual.is_some());
                        missing += usize::from(actual.is_none());
                    }
                }
            }
        }
    }
    assert!(successes > 0 && missing > 0);
}

#[test]
fn mapped_sampling_reuses_each_coordinate_once_and_keeps_failure_order() {
    let image = Rgb32FImage::from_pixel(64, 64, Rgb([0.4; 3]));
    let image_info = info(&image);
    let trace = Mutex::new(Vec::new());
    let mapping = |p: Point3<f64>| {
        trace.lock().unwrap().push((p.x.to_bits(), p.y.to_bits()));
        Some(Point2::new(p.x, p.y))
    };
    let mut source = sampler(&image, &image_info);
    source.native_mapping = Some(&mapping);
    for hole in [None, Some((18, 18)), Some((22, 22))] {
        let mut mask = GrayImage::from_pixel(64, 64, Luma([255]));
        if let Some((x, y)) = hole {
            mask.put_pixel(x, y, Luma([0]));
        }
        let base = window(&image, &mask);
        let expected = scalar_pair(&base, &source, 20.0, 20.0, true);
        let old_trace = std::mem::take(&mut *trace.lock().unwrap());
        let actual = native_patch_sample_pair(&base, &source, 20.0, 20.0, true);
        let new_trace = std::mem::take(&mut *trace.lock().unwrap());
        assert_eq!(pair_bits(actual), pair_bits(expected));
        assert!(old_trace.chunks_exact(2).all(|pair| pair[0] == pair[1]));
        assert_eq!(
            new_trace,
            old_trace.iter().step_by(2).copied().collect::<Vec<_>>()
        );
        match hole {
            None => {
                assert_eq!(old_trace.len(), 8);
                assert_eq!(new_trace.len(), 4);
            }
            Some((18, 18)) => assert!(new_trace.is_empty()),
            Some(_) => {
                assert_eq!(old_trace.len(), 6);
                assert_eq!(new_trace.len(), 3);
            }
        }
    }
}

#[test]
fn support_reuse_visits_only_the_exact_union_in_first_occurrence_order() {
    let supports = [(1, 4, 1, 4), (2, 5, 1, 4), (1, 4, 2, 5), (2, 5, 2, 5)];
    let mut expected = Vec::new();
    for (l, r, t, b) in supports {
        for y in t..=b {
            for x in l..=r {
                if !expected.contains(&(x, y)) {
                    expected.push((x, y));
                }
            }
        }
    }
    let mut seen = ValidatedSupports::default();
    let mut actual = Vec::new();
    for support in supports {
        seen.validate(support, |x, y| {
            actual.push((x, y));
            Some(())
        })
        .unwrap();
    }
    assert_eq!(actual, expected);
    assert_eq!(
        actual.len(),
        25,
        "four ordinary cubic footprints used to validate 64 pixels"
    );

    let mut seen = ValidatedSupports::default();
    seen.validate((1, 2, 1, 2), |_, _| Some(())).unwrap();
    assert!(
        seen.validate((8, 9, 8, 9), |x, y| ((x, y) != (9, 9)).then_some(()))
            .is_none()
    );
    assert_eq!(
        seen.len, 1,
        "a partially checked rectangle must not enter the cache"
    );
}

#[test]
fn disjoint_source_support_does_not_validate_the_bounding_box_gap() {
    let base_image = Rgb32FImage::from_pixel(64, 64, Rgb([0.4; 3]));
    let mask = GrayImage::from_pixel(64, 64, Luma([255]));
    let base = window(&base_image, &mask);
    let mapping = |p: Point3<f64>| {
        Some(Point2::new(
            if p.x < 32.0 { 16.25 } else { 48.25 },
            if p.y < 32.0 { 16.25 } else { 48.25 },
        ))
    };
    for poison in [(32, 32), (16, 16)] {
        let mut image = base_image.clone();
        image.put_pixel(poison.0, poison.1, Rgb([f32::NAN; 3]));
        let image_info = info(&image);
        let mut source = sampler(&image, &image_info);
        source.native_mapping = Some(&mapping);
        let expected = scalar_pair(&base, &source, 32.0, 32.0, true);
        let actual = native_patch_sample_pair(&base, &source, 32.0, 32.0, true);
        assert_eq!(pair_bits(actual), pair_bits(expected));
        assert_eq!(actual.is_some(), poison == (32, 32));
    }
}

#[test]
fn reused_support_retains_capture_border_and_nonfinite_rejections() {
    let base_image = Rgb32FImage::from_pixel(64, 64, Rgb([0.4; 3]));
    let mask = GrayImage::from_pixel(64, 64, Luma([255]));
    let base = window(&base_image, &mask);
    for value in [0.0, -0.0, f32::from_bits(1), f32::NAN, f32::INFINITY] {
        let mut image = base_image.clone();
        image.put_pixel(10, 50, Rgb([value; 3]));
        let image_info = info(&image);
        let source = sampler(&image, &image_info);
        let expected = scalar_pair(&base, &source, 12.0, 50.0, true);
        let actual = native_patch_sample_pair(&base, &source, 12.0, 50.0, true);
        assert_eq!(pair_bits(actual), pair_bits(expected));
        assert!(actual.is_none());
    }
}

fn measurement_bits(value: Option<([f64; 2], f64, f64, f64)>) -> Option<[u64; 5]> {
    value.map(|(d, c, r, g)| {
        [
            d[0].to_bits(),
            d[1].to_bits(),
            c.to_bits(),
            r.to_bits(),
            g.to_bits(),
        ]
    })
}

fn focal_bits(value: Option<intra_station::FocalPlaneMatch>) -> Option<([u64; 4], bool)> {
    value.map(|focal| {
        (
            [
                focal.matched_variance.to_bits(),
                focal.sharpness_ratio_before.to_bits(),
                focal.sharpness_ratio_matched.to_bits(),
                focal.sharpness_ratio_after.to_bits(),
            ],
            focal.capped,
        )
    })
}

#[test]
fn reused_prefilter_keeps_measurements_and_neighbour_accepted_set_identical() {
    let base_image = grain(17);
    let shifted = Rgb32FImage::from_fn(256, 256, |x, y| {
        get_high_quality_interpolated_pixel(&base_image, f64::from(x) - 3.25, f64::from(y) + 1.25)
    });
    let shifted = image::imageops::blur(&shifted, 0.65);
    let image_info = info(&shifted);
    let source = sampler(&shifted, &image_info);
    let mut mask = GrayImage::from_pixel(256, 256, Luma([255]));
    mask.put_pixel(30, 30, Luma([0]));
    let base = window(&base_image, &mask);
    let mode = NativeRefineMode::IntraStation {
        anchor_long_side: 9_504,
    };
    let mut old_grid = intra_station::ControlPointGrid::new(3, 3);
    let mut new_grid = intra_station::ControlPointGrid::new(3, 3);
    let mut old_set = Vec::new();
    let mut new_set = Vec::new();
    for (row, y) in [84, 116, 148].into_iter().enumerate() {
        for (column, x) in [84, 116, 148].into_iter().enumerate() {
            let expected = measure_native_control_point_with_sampling(
                &base,
                &source,
                x,
                y,
                2.0,
                mode,
                intra_station::INTRA_STATION_MATCH_RADIUS,
                intra_station::INTRA_STATION_SEARCH_SAMPLES,
                scalar_pair,
            );
            let actual = measure_native_control_point(
                &base,
                &source,
                x,
                y,
                2.0,
                mode,
                intra_station::INTRA_STATION_MATCH_RADIUS,
                intra_station::INTRA_STATION_SEARCH_SAMPLES,
            );
            assert_eq!(actual.covered, expected.covered);
            assert_eq!(actual.evaluated, expected.evaluated);
            assert_eq!(actual.rejection, expected.rejection);
            assert_eq!(actual.reverse_attempted, expected.reverse_attempted);
            assert_eq!(actual.round_trip_closed, expected.round_trip_closed);
            assert_eq!(
                focal_bits(actual.focal_plane),
                focal_bits(expected.focal_plane)
            );
            assert_eq!(
                actual.round_trip_px.to_bits(),
                expected.round_trip_px.to_bits()
            );
            assert_eq!(
                actual.best_correlation.map(f64::to_bits),
                expected.best_correlation.map(f64::to_bits)
            );
            assert_eq!(
                actual.peak_margin.map(f64::to_bits),
                expected.peak_margin.map(f64::to_bits)
            );
            assert_eq!(
                measurement_bits(actual.measured),
                measurement_bits(expected.measured)
            );
            assert_eq!(
                measurement_bits(actual.accepted),
                measurement_bits(expected.accepted)
            );
            for (point, grid, accepted) in [
                (expected, &mut old_grid, &mut old_set),
                (actual, &mut new_grid, &mut new_set),
            ] {
                if let Some((d, _, _, _)) = point.accepted
                    && intra_station::neighbour_median_consistent(
                        d,
                        grid.neighbour_median(column, row),
                    )
                {
                    grid.accept(column, row, d);
                    accepted.push((column, row, d.map(f64::to_bits)));
                }
            }
        }
    }
    assert!(
        !new_set.is_empty(),
        "the comparison must include real accepted correspondences"
    );
    assert!(
        new_set.len() < 9,
        "the mask hole must also exercise rejection"
    );
    assert_eq!(new_set, old_set);
}
