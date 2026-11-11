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
use std::collections::BTreeMap;

use super::{NativeReferenceMap, NativeSourceWarp, QualitySource, SourceLoader, project};

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
            let mapped = prefiltered_reference_point(mapped, source, native_registration)?;
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
            let pixel = native_registration
                .and_then(|native| native.mosaic_tone.as_ref())
                .map(|tone| tone.apply(world, pixel))
                .unwrap_or(pixel);
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

/// Build a station-level reference ROI from the actual final raw-owner plane.
///
/// A focus station may switch between several Source_RAW members at a much
/// finer scale than the final station geometry.  In that case a single raw
/// source cannot represent the whole ROI.  This helper samples each pixel
/// through the Source_RAW owner selected by the final ownership plane, while
/// requiring every participating owner to belong to the ROI's one station.
/// The caller may provide the already decoded origin owner to avoid a second
/// decode; additional owners are retained in `images` for later ROIs, capped
/// at 512 MiB. An oversized member is sampled once and dropped after its ROI
/// pixels; it never bypasses that cache budget by remaining in the map.
pub(crate) fn source_roi_for_station_quality(
    images: &mut BTreeMap<u16, Rgb32FImage>,
    primary_owner: u16,
    primary_image: Option<&Rgb32FImage>,
    ownership: &[u16],
    canvas_width: u32,
    roi: &QualityRoi,
    station_id: usize,
    sources: &BTreeMap<u16, &QualitySource>,
    residual: &ResidualWarp,
    load_source: &mut SourceLoader<'_>,
) -> Result<Rgb32FImage, &'static str> {
    source_roi_for_station_quality_with_budget(
        images,
        primary_owner,
        primary_image,
        ownership,
        canvas_width,
        roi,
        station_id,
        sources,
        residual,
        load_source,
        512 * 1024 * 1024,
    )
}

pub(super) fn source_roi_for_station_quality_with_budget(
    images: &mut BTreeMap<u16, Rgb32FImage>,
    primary_owner: u16,
    primary_image: Option<&Rgb32FImage>,
    ownership: &[u16],
    canvas_width: u32,
    roi: &QualityRoi,
    station_id: usize,
    sources: &BTreeMap<u16, &QualitySource>,
    residual: &ResidualWarp,
    load_source: &mut SourceLoader<'_>,
    max_cache_bytes: usize,
) -> Result<Rgb32FImage, &'static str> {
    if canvas_width == 0 || ownership.len() % canvas_width as usize != 0 {
        return Err(degradation::DIAGNOSTICS_ROI_INVALID);
    }
    let canvas_height = ownership.len() / canvas_width as usize;
    let right = roi
        .x
        .checked_add(roi.side)
        .ok_or(degradation::DIAGNOSTICS_ROI_INVALID)?;
    let bottom = roi
        .y
        .checked_add(roi.side)
        .ok_or(degradation::DIAGNOSTICS_ROI_INVALID)?;
    if right > canvas_width || bottom > u32::try_from(canvas_height).unwrap_or(u32::MAX) {
        return Err(degradation::DIAGNOSTICS_ROI_INVALID);
    }

    // Clear previous stations even when their members were loaded by a dense
    // ROI pass. More content-search windows must not retain all RAW frames.
    images.retain(|owner, _| {
        sources
            .get(owner)
            .is_some_and(|source| source.geometry.station_id == station_id)
    });
    let bytes = |image: &Rgb32FImage| {
        image
            .as_raw()
            .len()
            .saturating_mul(std::mem::size_of::<f32>())
    };
    let mut resident_bytes: usize = images.values().map(&bytes).sum();
    while resident_bytes > max_cache_bytes {
        let Some((_, image)) = images.pop_first() else {
            break;
        };
        resident_bytes = resident_bytes.saturating_sub(bytes(&image));
    }
    let mut pixels_by_owner = BTreeMap::<u16, Vec<(u32, u32)>>::new();
    for y in 0..roi.side {
        for x in 0..roi.side {
            let index = ((roi.y + y) * canvas_width + roi.x + x) as usize;
            let owner = ownership
                .get(index)
                .copied()
                .filter(|owner| *owner != 0)
                .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
            let source_info = sources
                .get(&owner)
                .copied()
                .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
            if source_info.geometry.station_id != station_id {
                return Err(degradation::OWNER_SOURCE_UNDECODABLE);
            }
            pixels_by_owner.entry(owner).or_default().push((x, y));
        }
    }
    let mut reference = Rgb32FImage::new(roi.side, roi.side);
    for (owner, pixels) in pixels_by_owner {
        let source_info = sources[&owner];
        let mut transient = None;
        let image = if owner == primary_owner {
            primary_image.ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?
        } else {
            if !images.contains_key(&owner) {
                let image =
                    load_source(source_info).map_err(|_| degradation::OWNER_SOURCE_UNDECODABLE)?;
                let image_bytes = bytes(&image);
                if image_bytes > max_cache_bytes {
                    transient = Some(image);
                } else {
                    while resident_bytes.saturating_add(image_bytes) > max_cache_bytes {
                        let Some((_, evicted)) = images.pop_first() else {
                            break;
                        };
                        resident_bytes = resident_bytes.saturating_sub(bytes(&evicted));
                    }
                    resident_bytes = resident_bytes.saturating_add(image_bytes);
                    images.insert(owner, image);
                }
            }
            transient
                .as_ref()
                .or_else(|| images.get(&owner))
                .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?
        };
        for (x, y) in pixels {
            let world = Point2::new(
                roi.world_origin.0 + f64::from(x),
                roi.world_origin.1 + f64::from(y),
            );
            let pixel = sample_quality_source_at_world(image, world, source_info, residual)?;
            reference.put_pixel(x, y, pixel);
        }
    }
    Ok(reference)
}

/// Measure output-to-source scale by following the final raw owner at each
/// probe centre. A station-level ROI may cross several raw members, so a
/// representative source's homography is not valid for the whole window.
pub(crate) fn measure_station_local_scale(
    roi: &QualityRoi,
    ownership: &[u16],
    canvas_width: u32,
    station_id: usize,
    sources: &BTreeMap<u16, &QualitySource>,
    residual: &ResidualWarp,
    acceptance_render_scale: f64,
) -> Result<quality_gate::LocalScaleMeasurement, &'static str> {
    if roi.side == 0
        || canvas_width == 0
        || ownership.len() % canvas_width as usize != 0
        || !acceptance_render_scale.is_finite()
        || acceptance_render_scale <= 0.0
    {
        return Err(degradation::DIAGNOSTICS_ROI_INVALID);
    }
    let canvas_height = ownership.len() / canvas_width as usize;
    let mut scales = Vec::new();
    let mut expected = 0usize;
    for y in (0..roi.side).step_by(16) {
        for x in (0..roi.side).step_by(16) {
            expected += 1;
            let output_x = roi.x.saturating_add(x);
            let output_y = roi.y.saturating_add(y);
            if output_x >= canvas_width || output_y as usize >= canvas_height {
                continue;
            }
            let owner = ownership[output_y as usize * canvas_width as usize + output_x as usize];
            let Some(source) = sources.get(&owner).copied() else {
                continue;
            };
            if source.geometry.station_id != station_id {
                continue;
            }
            let Some(inverse) =
                (source.geometry.tile_to_world * source.geometry.member_to_anchor).try_inverse()
            else {
                continue;
            };
            let map = |world: Point2<f64>| {
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
            };
            let center = Point2::new(
                roi.world_origin.0 + f64::from(x),
                roi.world_origin.1 + f64::from(y),
            );
            let Ok(xp) = map(center + nalgebra::Vector2::new(0.5, 0.0)) else {
                continue;
            };
            let Ok(xm) = map(center - nalgebra::Vector2::new(0.5, 0.0)) else {
                continue;
            };
            let Ok(yp) = map(center + nalgebra::Vector2::new(0.0, 0.5)) else {
                continue;
            };
            let Ok(ym) = map(center - nalgebra::Vector2::new(0.0, 0.5)) else {
                continue;
            };
            let dx = xp - xm;
            let dy = yp - ym;
            let determinant = (dx.x * dy.y - dx.y * dy.x).abs();
            if determinant.is_finite() && determinant > 1e-12 {
                scales.push(1.0 / determinant.sqrt());
            }
        }
    }
    if scales.len() < expected.div_ceil(2) {
        return Err(degradation::OWNER_SOURCE_UNDECODABLE);
    }
    scales.sort_by(f64::total_cmp);
    let median = if scales.len() % 2 == 0 {
        (scales[scales.len() / 2 - 1] + scales[scales.len() / 2]) * 0.5
    } else {
        scales[scales.len() / 2]
    };
    Ok(quality_gate::LocalScaleMeasurement {
        median,
        fraction_at_least_095: scales.iter().filter(|&&v| v >= 0.95).count() as f64
            / scales.len() as f64,
        samples: scales.len(),
        diagnostic_render_scale: acceptance_render_scale < 1.0,
    })
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
    // Direct RAW registration is the provenance used by source_roi_for_quality
    // and Local_Scale. It takes priority when both legacy native_warp and the
    // current registration are present on a station member.
    if source_info.native_registration.is_none()
        && let Some(native_warp) = source_info.native_warp.as_ref()
    {
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
    let mapped = source_point_for_reference(
        world,
        geometry,
        residual,
        source_info.native_registration.as_ref(),
        &inverse,
    )?;
    let mapped =
        prefiltered_reference_point(mapped, source, source_info.native_registration.as_ref())?;
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
    Ok(source_info
        .native_registration
        .as_ref()
        .and_then(|native| native.mosaic_tone.as_ref())
        .map(|tone| tone.apply(world, pixel))
        .unwrap_or(pixel))
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

/// The same half-pixel pyramid centre and edge clamp used by LayerSampler.
fn prefiltered_reference_point(
    point: Point2<f64>,
    source: &Rgb32FImage,
    native: Option<&NativeReferenceMap>,
) -> Result<Point2<f64>, &'static str> {
    let divisor = native.map(|n| n.source_prefilter_divisor).unwrap_or(1.0);
    if !divisor.is_finite() || divisor < 1.0 {
        return Err(degradation::DIAGNOSTICS_ROI_INVALID);
    }
    if divisor == 1.0 {
        return Ok(point);
    }
    Ok(Point2::new(
        ((point.x + 0.5) / divisor - 0.5).clamp(0.0, f64::from(source.width().saturating_sub(1))),
        ((point.y + 0.5) / divisor - 0.5).clamp(0.0, f64::from(source.height().saturating_sub(1))),
    ))
}
