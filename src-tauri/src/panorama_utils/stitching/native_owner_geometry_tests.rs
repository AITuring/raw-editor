use super::*;
use crate::panorama_utils::mosaic::{Field, NativeResidualDisplacement};
use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::stitching::{
    FocusSourceSampling, FocusTileSource, FocusTileSources, Projection,
    get_high_quality_interpolated_pixel,
};
use image::{GrayImage, Rgb, Rgb32FImage};
use nalgebra::Matrix3;
use std::path::PathBuf;

fn info(index: usize, dimensions: (u32, u32)) -> ImageInfo {
    ImageInfo {
        id: index,
        filename: format!("/fixture/native-owner-{index}.NEF"),
        width: dimensions.0,
        height: dimensions.1,
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

fn original(index: usize, dimensions: (u32, u32)) -> FocusTileSource {
    FocusTileSource {
        image: Rgb32FImage::from_fn(dimensions.0, dimensions.1, |x, y| {
            let v = (index as f32 * 31.0 + x as f32 * 5.0 + y as f32 * 7.0) / 1024.0;
            Rgb([v, v * 0.71, v * 0.39])
        }),
        refinement: FocusSourceRefinement {
            source_sampling: Some(FocusSourceSampling::for_test(
                &info(index, dimensions),
                &Matrix3::identity(),
                Projection::Planar,
                None,
            )),
            ..Default::default()
        },
    }
}

fn sources(cells: &[(u32, u32, u16)], count: usize) -> FocusTileSources {
    let mut owners = vec![0; 32 * 32];
    for &(x, y, owner) in cells {
        owners[(y * 32 + x) as usize] = owner;
    }
    FocusTileSources {
        ownership: OwnershipMap::new(
            32,
            32,
            owners,
            (0..count)
                .map(|index| PathBuf::from(info(index, (32, 32)).filename))
                .collect(),
        )
        .unwrap(),
        origin: (0.0, 0.0),
        originals: (0..count)
            .map(|index| Some(original(index, (32, 32))))
            .collect(),
    }
}

fn resolve(sources: &FocusTileSources, point: Point2<f64>) -> Option<NativeOwnerGeometry> {
    resolve_native_owner_geometry(point, &sources.ownership, sources.origin, |index| {
        let original = sources.originals[index].as_ref().unwrap();
        (&original.refinement, original.image.dimensions())
    })
}

/// Frozen pre-fallback implementation. Keep its winner selection, arithmetic,
/// coordinate mapping and RGB interpolation independent of the new resolver.
fn old_nearest_sample(
    sources: &FocusTileSources,
    point: Point2<f64>,
) -> Option<(u16, (u32, u32), Point2<f64>, Rgb<f32>)> {
    let (width, height) = sources.ownership.dimensions();
    let mut selected = None;
    let mut evidence_pixel = (0, 0);
    let mut minimum_distance = f64::INFINITY;
    let rounded = (point.x.round(), point.y.round());
    for dy in -1..=1 {
        for dx in -1..=1 {
            let x = rounded.0 + f64::from(dx);
            let y = rounded.1 + f64::from(dy);
            if x < 0.0 || y < 0.0 || x >= f64::from(width) || y >= f64::from(height) {
                continue;
            }
            let owner = sources.ownership.owner_at(x as u32, y as u32);
            let Some(index) = owner.checked_sub(1) else {
                continue;
            };
            let distance = (x - point.x).powi(2) + (y - point.y).powi(2);
            if distance < minimum_distance {
                minimum_distance = distance;
                selected = Some(usize::from(index));
                evidence_pixel = (x as u32, y as u32);
            }
        }
    }
    let index = selected?;
    let original = sources.originals.get(index)?.as_ref()?;
    let corrected = original.refinement.inverse_canvas_point(point);
    let source = original
        .refinement
        .source_sampling
        .as_ref()?
        .source_point(Point3::new(
            corrected.x + sources.origin.0,
            corrected.y + sources.origin.1,
            1.0,
        ))?;
    if source.x < 0.0
        || source.y < 0.0
        || source.x >= f64::from(original.image.width())
        || source.y >= f64::from(original.image.height())
    {
        return None;
    }
    Some((
        index as u16 + 1,
        evidence_pixel,
        source,
        get_high_quality_interpolated_pixel(&original.image, source.x, source.y),
    ))
}

#[test]
fn native_owner_geometry_falls_back_to_supported_owner_and_its_rgb() {
    let mut sources = sources(&[(8, 8, 1), (9, 8, 2)], 2);
    sources.originals[0]
        .as_mut()
        .unwrap()
        .refinement
        .translation = (40, 0);
    let point = Point2::new(8.25, 8.125);
    assert!(old_nearest_sample(&sources, point).is_none());
    let selected = resolve(&sources, point).unwrap();
    assert_eq!(selected.local_owner, 2);
    assert_eq!(selected.evidence_pixel, (9, 8));
    assert_eq!(selected.prepared_point, point);
    assert!(selected.used_fallback);
    let expected = get_high_quality_interpolated_pixel(
        &sources.originals[1].as_ref().unwrap().image,
        point.x,
        point.y,
    );
    assert_eq!(
        sources.sample(point).unwrap().0.map(f32::to_bits),
        expected.0.map(f32::to_bits)
    );
}

#[test]
fn native_owner_geometry_success_matches_old_sampling_bits_and_row_major_ties() {
    let cells: Vec<_> = (6..=12)
        .flat_map(|y| (6..=12).map(move |x| (x, y, ((x + 2 * y) % 3 + 1) as u16)))
        .collect();
    let mut sources = sources(&cells, 3);
    sources.origin = (0.25, -0.125);
    for (index, original) in sources.originals.iter_mut().enumerate() {
        let original = original.as_mut().unwrap();
        let mut field = Field::<2>::new(32, 32, 8.0);
        field.values.fill([0.125 * index as f64, -0.0625]);
        original.refinement.displacement = Some(NativeResidualDisplacement::from_test_field(field));
        original.refinement.source_sampling = Some(FocusSourceSampling::for_test(
            &info(index, (32, 32)),
            &Matrix3::new(
                1.0, 0.015625, 0.25, -0.03125, 1.0, -0.125, 0.0001, 0.0002, 1.0,
            ),
            Projection::Planar,
            None,
        ));
    }
    let phases = [0.0, 0.125, 0.25, 0.49, 0.5, 0.75, 0.875];
    for y in 8..=10 {
        for x in 8..=10 {
            for dy in phases {
                for dx in phases {
                    let point = Point2::new(f64::from(x) + dx, f64::from(y) + dy);
                    let (owner, pixel, prepared, rgb) =
                        old_nearest_sample(&sources, point).unwrap();
                    let selected = resolve(&sources, point).unwrap();
                    assert_eq!(selected.local_owner, owner, "point={point:?}");
                    assert_eq!(selected.evidence_pixel, pixel, "point={point:?}");
                    assert_eq!(selected.prepared_point.x.to_bits(), prepared.x.to_bits());
                    assert_eq!(selected.prepared_point.y.to_bits(), prepared.y.to_bits());
                    assert!(!selected.used_fallback);
                    assert_eq!(
                        sources.sample(point).unwrap().0.map(f32::to_bits),
                        rgb.0.map(f32::to_bits)
                    );
                }
            }
        }
    }
}

#[test]
fn native_owner_geometry_ties_use_pixel_order_after_distinct_owner_deduplication() {
    // Owner 1 is first encountered far away. Its later nearest cell ties with
    // owner 2, whose actual evidence cell is earlier in row-major order.
    let sources = sources(&[(8, 8, 1), (8, 9, 2), (9, 9, 1)], 2);
    let point = Point2::new(8.5, 9.0);
    let selected = resolve(&sources, point).unwrap();
    assert_eq!(selected.local_owner, 2);
    assert_eq!(selected.evidence_pixel, (8, 9));
    assert!(!selected.used_fallback);
    assert_eq!(old_nearest_sample(&sources, point).unwrap().0, 2);
}

#[test]
fn native_owner_geometry_preserves_origin_translation_residual_and_source_coordinates() {
    let mut sources = sources(&[(8, 10, 1)], 1);
    sources.origin = (7.5, -4.25);
    let original = sources.originals[0].as_mut().unwrap();
    let mut field = Field::<2>::new(32, 32, 8.0);
    field.values.fill([0.25, -0.5]);
    original.refinement = FocusSourceRefinement {
        translation: (2, -3),
        displacement: Some(NativeResidualDisplacement::from_test_field(field)),
        source_sampling: Some(FocusSourceSampling::for_test(
            &info(0, (32, 32)),
            &Matrix3::new(1.0, 0.0, 3.25, 0.0, 1.0, -2.5, 0.0, 0.0, 1.0),
            Projection::Planar,
            None,
        )),
        ..Default::default()
    };
    let point = Point2::new(8.25, 9.5);
    let selected = resolve(&sources, point).unwrap();
    assert_eq!(selected.prepared_point, Point2::new(10.75, 10.25));
    assert_eq!(selected.evidence_pixel, (8, 10));
    assert!(!selected.used_fallback);
    let old = old_nearest_sample(&sources, point).unwrap();
    assert_eq!(
        sources.sample(point).unwrap().0.map(f32::to_bits),
        old.3.0.map(f32::to_bits)
    );
}

#[test]
fn native_owner_geometry_rejects_calibrated_padding_inside_rectangular_source_bounds() {
    let metadata = [
        ("Make", "NIKON CORPORATION"),
        ("Model", "NIKON Z 7_2"),
        ("LensModel", "NIKKOR Z 24-120mm f/4 S"),
        ("FocalLength", "24"),
        ("FocalLengthIn35mmFilm", "24"),
    ]
    .into_iter()
    .map(|(key, value)| (key.into(), value.into()))
    .collect();
    let calibration =
        crate::lens_correction::stack_lens_calibration(&metadata, (8256, 5504)).unwrap();
    let image = Rgb32FImage::from_pixel(64, 48, Rgb([0.5; 3]));
    let corrected = calibration.correct(&image).unwrap();
    let dimensions = corrected.dimensions();
    let footprint = crate::lens_correction::StackLensFootprint {
        calibration,
        original: image.dimensions(),
        corrected: dimensions,
    };
    let (x, y) = (0..dimensions.1)
        .find_map(|y| {
            (0..dimensions.0 - 1)
                .find(|&x| !footprint.contains(f64::from(x), f64::from(y), dimensions))
                .map(|x| (x, y))
        })
        .expect("calibrated fixture must contain padding");
    let mut first_info = info(0, dimensions);
    first_info.source_footprint = Some(footprint);
    let mut first = original(0, dimensions);
    first.refinement.source_sampling = Some(FocusSourceSampling::for_test(
        &first_info,
        &Matrix3::identity(),
        Projection::Planar,
        None,
    ));
    let mut owners = vec![0; (dimensions.0 * dimensions.1) as usize];
    owners[(y * dimensions.0 + x) as usize] = 1;
    owners[(y * dimensions.0 + x + 1) as usize] = 2;
    let sources = FocusTileSources {
        ownership: OwnershipMap::new(
            dimensions.0,
            dimensions.1,
            owners,
            vec!["a.NEF".into(), "b.NEF".into()],
        )
        .unwrap(),
        origin: (0.0, 0.0),
        originals: vec![Some(first), Some(original(1, dimensions))],
    };
    let point = Point2::new(f64::from(x), f64::from(y));
    assert!(old_nearest_sample(&sources, point).is_none());
    let selected = resolve(&sources, point).unwrap();
    assert_eq!(selected.local_owner, 2);
    assert_eq!(selected.evidence_pixel, (x + 1, y));
    assert_eq!(selected.prepared_point, point);
    assert!(selected.used_fallback);
    assert!(sources.sample(point).is_some());
}

#[test]
fn native_owner_geometry_checks_each_distinct_owner_once_and_keeps_nearest_evidence() {
    let mut sources = sources(&[(7, 7, 1), (8, 8, 1), (9, 8, 1), (9, 9, 2)], 2);
    sources.originals[0]
        .as_mut()
        .unwrap()
        .refinement
        .translation = (40, 0);
    let mut calls = [0; 2];
    let selected = resolve_native_owner_geometry(
        Point2::new(8.0, 8.0),
        &sources.ownership,
        sources.origin,
        |index| {
            calls[index] += 1;
            let original = sources.originals[index].as_ref().unwrap();
            (&original.refinement, original.image.dimensions())
        },
    )
    .unwrap();
    assert_eq!(calls, [1, 1]);
    assert_eq!(selected.local_owner, 2);
    assert!(selected.used_fallback);
    sources.originals[0]
        .as_mut()
        .unwrap()
        .refinement
        .translation = (0, 0);
    let selected = resolve(&sources, Point2::new(8.0, 8.0)).unwrap();
    assert_eq!(selected.evidence_pixel, (8, 8));
    assert!(!selected.used_fallback);
}

#[test]
fn native_owner_geometry_does_not_expand_stencil_or_accept_unsupported_candidates() {
    // Owner 3 is supported but lies two cells away from the rounded point.
    let mut sources = sources(&[(8, 8, 1), (9, 8, 2), (10, 8, 3)], 3);
    for original in &mut sources.originals[..2] {
        original.as_mut().unwrap().refinement.translation = (40, 0);
    }
    let mut calls = [0; 3];
    assert!(
        resolve_native_owner_geometry(
            Point2::new(8.0, 8.0),
            &sources.ownership,
            sources.origin,
            |index| {
                calls[index] += 1;
                let original = sources.originals[index].as_ref().unwrap();
                (&original.refinement, original.image.dimensions())
            }
        )
        .is_none()
    );
    assert_eq!(calls, [1, 1, 0]);
    assert!(sources.sample(Point2::new(8.0, 8.0)).is_none());
    for point in [
        Point2::new(20.0, 20.0),
        Point2::new(-5.0, 4.0),
        Point2::new(f64::NAN, 4.0),
        Point2::new(f64::INFINITY, 4.0),
    ] {
        assert!(
            resolve_native_owner_geometry(point, &sources.ownership, sources.origin, |_| panic!(
                "no candidate must access metadata"
            ))
            .is_none()
        );
    }
}

#[test]
fn native_owner_geometry_respects_actual_prepared_buffer_bounds() {
    let sources = sources(&[(8, 8, 1), (9, 8, 2)], 2);
    let selected = resolve_native_owner_geometry(
        Point2::new(8.0, 8.0),
        &sources.ownership,
        sources.origin,
        |index| {
            let original = sources.originals[index].as_ref().unwrap();
            // Even when metadata permits the point, a smaller actual buffer does
            // not. Production validates dimensions before entering the pixel loop.
            (
                &original.refinement,
                if index == 0 {
                    (8, 8)
                } else {
                    original.image.dimensions()
                },
            )
        },
    )
    .unwrap();
    assert_eq!(selected.local_owner, 2);
    assert!(selected.used_fallback);
}

#[test]
fn native_owner_geometry_keeps_supported_black_owner_without_rgb_fallback() {
    let mut sources = sources(&[(8, 8, 1), (9, 8, 2)], 2);
    sources.originals[0].as_mut().unwrap().image = Rgb32FImage::from_pixel(32, 32, Rgb([0.0; 3]));
    sources.originals[1].as_mut().unwrap().image = Rgb32FImage::from_pixel(32, 32, Rgb([0.8; 3]));
    let point = Point2::new(8.0, 8.0);
    let selected = resolve(&sources, point).unwrap();
    assert_eq!(selected.local_owner, 1);
    assert!(!selected.used_fallback);
    assert_eq!(
        sources.sample(point).unwrap().0.map(f32::to_bits),
        [0.0_f32.to_bits(); 3]
    );
    assert_eq!(old_nearest_sample(&sources, point).unwrap().0, 1);
}
