//! The survey of every frame a focus scan takes before composing.
//!
//! Each frame is loaded once and measured three ways on the canvas: its
//! detail on the grid the composition decisions use and on the finer grid the
//! detail repair compares the finished canvas with (see `soft_frames`,
//! `detail_repair`), and its tone on the coarse grid of the joint exposure
//! solve (see `frame_exposure`).

use super::super::stitching::Projection;
use super::super::stitching::transformed_image_region;
use crate::panorama_utils::shared::ImageInfo;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use std::collections::HashMap;
use tauri::AppHandle;
use tauri::Emitter;
use tauri::Runtime;

use super::{
    DetailGrid, ExposureGrid, Field, FrameDetail, GRID_STEP, LayerSampler,
    STREAMING_ANALYSIS_LONG_SIDE, measure_frame_detail, planar_canvas_scale,
    prefilter_minified_source, sample_frame_tone,
};

/// Every frame's measurements, in input order.
pub(super) struct FrameSurvey {
    pub(super) details: Vec<FrameDetail>,
    pub(super) repair_details: Vec<FrameDetail>,
    pub(super) tones: Vec<Vec<(usize, [f64; 3])>>,
}

/// Measures every frame of `images` (see the module documentation); `grids`
/// are the composition, repair and exposure grids.
#[allow(clippy::too_many_arguments)]
pub(super) fn survey_frames<R: Runtime, F>(
    images: &[&ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    (offset_x, offset_y): (f64, f64),
    (width, height): (u32, u32),
    (grid, repair_grid, exposure_grid): (&DetailGrid, &DetailGrid, &ExposureGrid),
    app: &AppHandle<R>,
    event: &str,
    load: &mut F,
) -> Result<FrameSurvey, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    let mut details = Vec::with_capacity(images.len());
    let mut repair_details = Vec::with_capacity(images.len());
    let mut tones = Vec::with_capacity(images.len());
    for (index, &info) in images.iter().enumerate() {
        let _ = app.emit(
            event,
            format!(
                "Measuring the detail of source {} of {}",
                index + 1,
                images.len()
            ),
        );
        let transform = &homographies[&info.id];
        let Some((left, right, top, bottom)) = transformed_image_region(
            info, transform, projection, offset_x, offset_y, width, height,
        ) else {
            details.push(FrameDetail::default());
            repair_details.push(FrameDetail::default());
            tones.push(Vec::new());
            continue;
        };
        let (source, source_divisor) =
            prefilter_minified_source(load(info)?, planar_canvas_scale(info, transform));
        let layer_width = right - left + 1;
        let layer_height = bottom - top + 1;
        let layer_scale =
            (layer_width.max(layer_height) as f64 / STREAMING_ANALYSIS_LONG_SIDE as f64).max(1.0);
        let sampler = LayerSampler {
            info,
            source: &source,
            source_divisor,
            inverse: transform
                .try_inverse()
                .ok_or("The stack contains a non-invertible alignment.")?,
            native_mapping: None,
            projection,
            offset: (offset_x, offset_y),
            left,
            top,
            scale: layer_scale,
            residual: Field::new(
                (layer_width as f64 / layer_scale).ceil() as u32,
                (layer_height as f64 / layer_scale).ceil() as u32,
                GRID_STEP as f64,
            ),
        };
        details.push(measure_frame_detail(
            &sampler,
            (left, right, top, bottom),
            grid,
        ));
        repair_details.push(measure_frame_detail(
            &sampler,
            (left, right, top, bottom),
            repair_grid,
        ));
        tones.push(sample_frame_tone(
            &sampler,
            (left, right, top, bottom),
            exposure_grid,
        ));
    }
    Ok(FrameSurvey {
        details,
        repair_details,
        tones,
    })
}
