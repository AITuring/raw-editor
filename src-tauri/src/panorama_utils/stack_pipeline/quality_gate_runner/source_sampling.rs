//! Source_RAW reference sampling for Quality_Gate ROIs.
//!
//! Maps output ROIs back to each owner source through registration, native
//! station layers and the run tone transform, and samples the reference ROI.

use super::super::degradation;
use super::super::quality_gate;
use super::super::quality_gate::QualityRoi;
use super::super::quality_gate::SourceGeometry;
use super::super::residual_warp::ResidualWarp;
use image::Rgb;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use nalgebra::Point2;

use super::{NativeReferenceMap, NativeSourceWarp, QualitySource, project};

pub(super) fn source_quad(source: &QualitySource) -> Option<[Point2<f64>; 4]> {
    let matrix = source.geometry.tile_to_world * source.geometry.member_to_anchor;
    let (w, h) = source.dimensions;
    let corners = [
        Point2::new(0.0, 0.0),
        Point2::new(f64::from(w.saturating_sub(1)), 0.0),
        Point2::new(
            f64::from(w.saturating_sub(1)),
            f64::from(h.saturating_sub(1)),
        ),
        Point2::new(0.0, f64::from(h.saturating_sub(1))),
    ];
    Some([
        project(&matrix, corners[0])?,
        project(&matrix, corners[1])?,
        project(&matrix, corners[2])?,
        project(&matrix, corners[3])?,
    ])
}

/// Source sampling through the complete output-to-source mapping, using the
/// same Catmull-Rom kernel as rendering so reference noise is not artificially
/// reduced by a different bilinear filter.
/// This is the single allowed reference resample; output pixels are copied as
/// is by [`output_roi`].
pub(super) fn source_point_for_reference(
    world: Point2<f64>,
    geometry: &SourceGeometry,
    residual: &ResidualWarp,
    native_registration: Option<&NativeReferenceMap>,
    inverse: &Matrix3<f64>,
) -> Result<Point2<f64>, &'static str> {
    if let Some(native) = native_registration {
        let unwarped = residual.warp_inverse(world, geometry.station_id);
        if let Some(sampling) = native.refinement.source_sampling.as_ref() {
            let station = native
                .corrected_station(unwarped)
                .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
            Ok(sampling
                .source_point(nalgebra::Point3::new(
                    station.x + native.station_origin.0,
                    station.y + native.station_origin.1,
                    1.0,
                ))
                .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?)
        } else {
            let corrected = native
                .reference_world(unwarped)
                .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
            project(inverse, corrected).ok_or(degradation::OWNER_SOURCE_UNDECODABLE)
        }
    } else {
        quality_gate::map_world_to_source_with_inverse(
            world,
            geometry.station_id,
            residual,
            inverse,
        )
    }
}

pub(crate) fn measure_source_local_scale(
    roi: &QualityRoi,
    source: &QualitySource,
    residual: &ResidualWarp,
    render_scale: f64,
) -> Result<quality_gate::LocalScaleMeasurement, &'static str> {
    let inverse = (source.geometry.tile_to_world * source.geometry.member_to_anchor)
        .try_inverse()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    quality_gate::local_scale_with_mapper(roi, render_scale, |world| {
        if source.native_registration.is_some() {
            source_point_for_reference(
                world,
                &source.geometry,
                residual,
                source.native_registration.as_ref(),
                &inverse,
            )
        } else {
            map_world_to_source_for_quality(
                world,
                &source.geometry,
                residual,
                &inverse,
                source.native_warp.as_ref(),
            )
        }
    })
}

pub(super) fn source_roi(
    source: &Rgb32FImage,
    roi: &QualityRoi,
    geometry: &SourceGeometry,
    residual: &ResidualWarp,
) -> Result<Rgb32FImage, &'static str> {
    source_roi_with_native(source, roi, geometry, residual, None)
}

pub(super) fn source_roi_with_registration(
    source: &Rgb32FImage,
    roi: &QualityRoi,
    geometry: &SourceGeometry,
    residual: &ResidualWarp,
    native_registration: Option<&NativeReferenceMap>,
) -> Result<Rgb32FImage, &'static str> {
    let inverse = (geometry.tile_to_world * geometry.member_to_anchor)
        .try_inverse()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let mut result = Rgb32FImage::new(roi.side, roi.side);
    for y in 0..roi.side {
        for x in 0..roi.side {
            let world = Point2::new(
                roi.world_origin.0 + f64::from(x),
                roi.world_origin.1 + f64::from(y),
            );
            let mapped = source_point_for_reference(
                world,
                geometry,
                residual,
                native_registration,
                &inverse,
            )?;
            let max_x = f64::from(source.width().saturating_sub(1));
            let max_y = f64::from(source.height().saturating_sub(1));
            if !mapped.x.is_finite()
                || !mapped.y.is_finite()
                || mapped.x < 0.0
                || mapped.y < 0.0
                || mapped.x > max_x
                || mapped.y > max_y
            {
                return Err(degradation::OWNER_SOURCE_UNDECODABLE);
            }
            let pixel = crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
                source, mapped.x, mapped.y,
            );
            result.put_pixel(x, y, pixel);
        }
    }
    Ok(result)
}

/// Select the reference kernel from the provenance of the actual rendered payload.
pub(super) fn source_roi_for_quality(
    image: &Rgb32FImage,
    roi: &QualityRoi,
    source: &QualitySource,
    residual: &ResidualWarp,
) -> Result<Rgb32FImage, &'static str> {
    if source.native_registration.is_some() {
        source_roi_with_registration(
            image,
            roi,
            &source.geometry,
            residual,
            source.native_registration.as_ref(),
        )
    } else {
        source_roi_with_native(
            image,
            roi,
            &source.geometry,
            residual,
            source.native_warp.as_ref(),
        )
    }
}

/// Replay the bounded per-station tone transform that was actually applied by
/// the owner compositor before the Quality_Gate compares output to its source
/// reference.  The compositor corrects only the low-frequency band, so this
/// whole-ROI affine replay is intentionally used for the dense mean/gradient
/// evidence only; it prevents an already-applied exposure correction from
/// being reported as a source-fidelity defect without changing any threshold.
pub(super) fn apply_run_tone_transform(reference: &mut Rgb32FImage, station_id: usize) {
    let Some(report) = crate::panorama_utils::stack_pipeline::tone::run_report_snapshot() else {
        return;
    };
    if matches!(
        report.status,
        crate::panorama_utils::stack_pipeline::report::ToneStatus::Identity
            | crate::panorama_utils::stack_pipeline::report::ToneStatus::NotRun
    ) {
        return;
    }
    let Some(tile) = report
        .tiles
        .iter()
        .find(|tile| tile.station_index == station_id)
    else {
        return;
    };
    for pixel in reference.pixels_mut() {
        for channel in 0..3 {
            pixel[channel] = (tile.gain[channel] as f32 * pixel[channel]
                + tile.offset[channel] as f32)
                .clamp(0.0, 1.0);
        }
    }
}

/// Sample a Source_RAW through the exact station-local native refinement that
/// was applied to the owner layer, when one exists.  The regular geometry
/// remains the fallback for fixtures and legacy paths that did not run native
/// refinement.
pub(super) fn source_roi_with_native(
    source: &Rgb32FImage,
    roi: &QualityRoi,
    geometry: &SourceGeometry,
    residual: &ResidualWarp,
    native_warp: Option<&NativeSourceWarp>,
) -> Result<Rgb32FImage, &'static str> {
    let inverse = (geometry.tile_to_world * geometry.member_to_anchor)
        .try_inverse()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let station_inverse = native_warp.and_then(|warp| warp.station_to_world.try_inverse());
    let tile_to_source = native_warp.and_then(|warp| {
        let source_to_tile = warp.station_to_world.try_inverse()?
            * geometry.tile_to_world
            * geometry.member_to_anchor;
        source_to_tile.try_inverse()
    });
    let mut result = Rgb32FImage::new(roi.side, roi.side);
    for y in 0..roi.side {
        for x in 0..roi.side {
            let world = Point2::new(
                roi.world_origin.0 + f64::from(x),
                roi.world_origin.1 + f64::from(y),
            );
            if let (Some(station_inverse), Some(tile_to_source), Some(native_warp)) = (
                station_inverse.as_ref(),
                tile_to_source.as_ref(),
                native_warp,
            ) {
                // `tile_local` is the coordinate in the already rendered
                // station plane.  Sampling that plane through the same 4x4
                // cubic kernel as the final compositor is what keeps a
                // gradient/MTF comparison from treating the raw input as a
                // sharper one-stage reference than the delivered pixels.
                let unwarped = residual.warp_inverse(world, geometry.station_id);
                let tile_local = project(station_inverse, unwarped)
                    .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
                let sampled = sample_station_layer_reference(
                    source,
                    tile_local,
                    tile_to_source,
                    native_warp.displacement.as_ref(),
                )
                .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
                result.put_pixel(x, y, sampled);
            } else {
                let mapped = map_world_to_source_for_quality(
                    world,
                    geometry,
                    residual,
                    &inverse,
                    native_warp,
                )?;
                let max_x = f64::from(source.width().saturating_sub(1));
                let max_y = f64::from(source.height().saturating_sub(1));
                if !mapped.x.is_finite()
                    || !mapped.y.is_finite()
                    || mapped.x < 0.0
                    || mapped.y < 0.0
                    || mapped.x > max_x
                    || mapped.y > max_y
                {
                    return Err(degradation::OWNER_SOURCE_UNDECODABLE);
                }
                result.put_pixel(
                    x,
                    y,
                    crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
                        source, mapped.x, mapped.y,
                    ),
                );
            }
        }
    }
    Ok(result)
}

pub(crate) fn sample_quality_source_at_world(
    source: &Rgb32FImage,
    world: Point2<f64>,
    source_info: &QualitySource,
    residual: &ResidualWarp,
) -> Result<Rgb<f32>, &'static str> {
    let geometry = &source_info.geometry;
    if let Some(native_warp) = source_info.native_warp.as_ref() {
        let station_inverse = native_warp
            .station_to_world
            .try_inverse()
            .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
        let source_to_tile = station_inverse * geometry.tile_to_world * geometry.member_to_anchor;
        let tile_to_source = source_to_tile
            .try_inverse()
            .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
        let unwarped = residual.warp_inverse(world, geometry.station_id);
        let tile_local =
            project(&station_inverse, unwarped).ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
        return sample_station_layer_reference(
            source,
            tile_local,
            &tile_to_source,
            native_warp.displacement.as_ref(),
        )
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE);
    }
    let inverse = (geometry.tile_to_world * geometry.member_to_anchor)
        .try_inverse()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let mapped = map_world_to_source_for_quality(world, geometry, residual, &inverse, None)?;
    let max_x = f64::from(source.width().saturating_sub(1));
    let max_y = f64::from(source.height().saturating_sub(1));
    if !mapped.x.is_finite()
        || !mapped.y.is_finite()
        || mapped.x < 0.0
        || mapped.y < 0.0
        || mapped.x > max_x
        || mapped.y > max_y
    {
        return Err(degradation::OWNER_SOURCE_UNDECODABLE);
    }
    Ok(
        crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
            source, mapped.x, mapped.y,
        ),
    )
}

/// Read one integer station-layer pixel from the Source_RAW.  Native
/// refinement is a bilinear read of the pre-warp layer; the pre-warp layer
/// itself is the cubic Source_RAW sample produced by `render_focus_layer`.
pub(super) fn sample_station_layer_integer(
    source: &Rgb32FImage,
    tile_to_source: &Matrix3<f64>,
    tile_x: f64,
    tile_y: f64,
    displacement: Option<&crate::panorama_utils::mosaic::NativeResidualDisplacement>,
) -> Option<Rgb<f32>> {
    let sample_prewarp = |x: f64, y: f64| {
        let raw = project(tile_to_source, Point2::new(x, y))?;
        if !raw.x.is_finite()
            || !raw.y.is_finite()
            || raw.x < 0.0
            || raw.y < 0.0
            || raw.x > f64::from(source.width().saturating_sub(1))
            || raw.y > f64::from(source.height().saturating_sub(1))
        {
            return None;
        }
        Some(
            crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
                source, raw.x, raw.y,
            ),
        )
    };
    let Some(displacement) = displacement else {
        return sample_prewarp(tile_x, tile_y);
    };
    let [delta_x, delta_y] = displacement.at(tile_x, tile_y);
    let sample_x = tile_x + delta_x;
    let sample_y = tile_y + delta_y;
    if !sample_x.is_finite() || !sample_y.is_finite() || sample_x < 0.0 || sample_y < 0.0 {
        return None;
    }
    let x0 = sample_x.floor();
    let y0 = sample_y.floor();
    let fx = (sample_x - x0) as f32;
    let fy = (sample_y - y0) as f32;
    let top_left = sample_prewarp(x0, y0)?;
    let top_right = sample_prewarp(x0 + 1.0, y0)?;
    let bottom_left = sample_prewarp(x0, y0 + 1.0)?;
    let bottom_right = sample_prewarp(x0 + 1.0, y0 + 1.0)?;
    Some(Rgb(std::array::from_fn(|channel| {
        let top = top_left[channel] * (1.0 - fx) + top_right[channel] * fx;
        let bottom = bottom_left[channel] * (1.0 - fx) + bottom_right[channel] * fx;
        top * (1.0 - fy) + bottom * fy
    })))
}

/// Cubic sample of the station-layer values around one final output pixel.
/// The values are reconstructed from the exact source transform above; no
/// third image is decoded and no metric threshold is changed.
pub(super) fn sample_station_layer_reference(
    source: &Rgb32FImage,
    tile: Point2<f64>,
    tile_to_source: &Matrix3<f64>,
    displacement: Option<&crate::panorama_utils::mosaic::NativeResidualDisplacement>,
) -> Option<Rgb<f32>> {
    if !tile.x.is_finite() || !tile.y.is_finite() {
        return None;
    }
    let x_floor = tile.x.floor() as i32;
    let y_floor = tile.y.floor() as i32;
    let amount_x = tile.x - f64::from(x_floor);
    let amount_y = tile.y - f64::from(y_floor);
    if amount_x.abs() < 1.0e-9 && amount_y.abs() < 1.0e-9 {
        return sample_station_layer_integer(
            source,
            tile_to_source,
            f64::from(x_floor),
            f64::from(y_floor),
            displacement,
        );
    }
    let mut values = [[[0.0f64; 3]; 4]; 4];
    let mut local_min = [f64::INFINITY; 3];
    let mut local_max = [f64::NEG_INFINITY; 3];
    for (row, sample_y) in ((y_floor - 1)..=(y_floor + 2)).enumerate() {
        for (column, sample_x) in ((x_floor - 1)..=(x_floor + 2)).enumerate() {
            let pixel = sample_station_layer_integer(
                source,
                tile_to_source,
                f64::from(sample_x),
                f64::from(sample_y),
                displacement,
            )?;
            for channel in 0..3 {
                let value = f64::from(pixel[channel]);
                values[row][column][channel] = value;
                local_min[channel] = local_min[channel].min(value);
                local_max[channel] = local_max[channel].max(value);
            }
        }
    }
    Some(Rgb(std::array::from_fn(|channel| {
        let mut rows = [0.0f64; 4];
        for row in 0..4 {
            rows[row] = cubic_sample(
                values[row][0][channel],
                values[row][1][channel],
                values[row][2][channel],
                values[row][3][channel],
                amount_x,
            );
        }
        cubic_sample(rows[0], rows[1], rows[2], rows[3], amount_y)
            .clamp(local_min[channel], local_max[channel]) as f32
    })))
}

pub(super) fn cubic_sample(p0: f64, p1: f64, p2: f64, p3: f64, amount: f64) -> f64 {
    p1 + 0.5
        * amount
        * (p2 - p0
            + amount * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + amount * (3.0 * (p1 - p2) + p3 - p0)))
}

pub(super) fn map_world_to_source_for_quality(
    world: Point2<f64>,
    geometry: &SourceGeometry,
    residual: &ResidualWarp,
    inverse: &Matrix3<f64>,
    native_warp: Option<&NativeSourceWarp>,
) -> Result<Point2<f64>, &'static str> {
    let unwarped = residual.warp_inverse(world, geometry.station_id);
    let Some(native_warp) = native_warp else {
        return project(inverse, unwarped).map_or(Err(degradation::OWNER_SOURCE_UNDECODABLE), Ok);
    };
    let tile_inverse = native_warp
        .station_to_world
        .try_inverse()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let tile_local =
        project(&tile_inverse, unwarped).ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    // `NativeResidualDisplacement` is addressed in the rendered station-tile
    // canvas.  `station_to_world.inverse()` already returned that coordinate;
    // adding `sampling_origin` here converted it back to world space and made
    // the field lookup depend on the tile's global offset.  The layer warp
    // applies the same field at `(candidate.left + x, candidate.top + y)`, so
    // use the tile-local point directly and keep the source transform in the
    // world frame below.
    let delta = native_warp
        .displacement
        .as_ref()
        .map(|displacement| displacement.at(tile_local.x, tile_local.y))
        .unwrap_or([0.0, 0.0]);
    let source_inverse = native_warp
        .source_to_station
        .try_inverse()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    project(
        &source_inverse,
        Point2::new(unwarped.x + delta[0], unwarped.y + delta[1]),
    )
    .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)
}
