//! Intra_Station_Registrar entry for the default station path (需求 2.2–2.9).
//!
//! Runs native refinement over an already rendered layer through the inverse
//! map that final RAW sampling uses and returns the accepted displacement in
//! canvas pixels.

use super::super::stack_pipeline::intra_station;
use super::super::stack_pipeline::report::IntraStationFrameRecord;
use super::super::stitching::Projection;
use crate::panorama_utils::shared::ImageInfo;
use image::GrayImage;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use nalgebra::Point2;
use nalgebra::Point3;

use super::{
    ANALYSIS_LONG_SIDE, Field, GRID_STEP, LayerSampler, NativeRefineMode,
    intra_station_frame_record, refine_layer, refine_native_layer,
};

/// The displacement field a native refinement left behind, addressed in canvas
/// pixels (需求 2.3–2.7).
///
/// The refinement measures and corrects in the layer's analysis units; a caller
/// that already holds the layer rendered on the canvas needs the same
/// correction in canvas pixels, which is what [`Self::at`] returns.
#[derive(Clone, Debug)]
pub(crate) struct NativeResidualDisplacement {
    pub(super) field: Field<2>,
    pub(super) left: u32,
    pub(super) top: u32,
    pub(super) scale: f64,
}

impl NativeResidualDisplacement {
    #[cfg(test)]
    pub(crate) fn from_test_field(field: Field<2>) -> Self {
        Self {
            field,
            left: 0,
            top: 0,
            scale: 1.0,
        }
    }

    /// Canvas-pixel offset to add to a canvas position before sampling the
    /// layer, i.e. exactly what [`LayerSampler::source_point`] adds.
    pub(crate) fn at(&self, canvas_x: f64, canvas_y: f64) -> [f64; 2] {
        let delta = self.field.at(
            (canvas_x - f64::from(self.left)) / self.scale,
            (canvas_y - f64::from(self.top)) / self.scale,
        );
        [delta[0] * self.scale, delta[1] * self.scale]
    }

    /// Largest correction the field carries, in canvas pixels.  Reported so a
    /// run log shows whether the refinement moved anything at all.
    pub(crate) fn max_offset_px(&self) -> f64 {
        self.field
            .values
            .iter()
            .map(|delta| delta[0].hypot(delta[1]) * self.scale)
            .fold(0.0, f64::max)
    }
}

/// One frame of the Intra_Station_Registrar on the default station path
/// (需求 2.2–2.9).
pub(crate) struct StationFrameRegistration {
    /// The `intra_station[*].frames[*]` entry of 需求 2.9.
    pub(crate) record: IntraStationFrameRecord,
    /// The correction to apply to the already rendered layer, or `None` when
    /// nothing was measured.
    pub(crate) displacement: Option<NativeResidualDisplacement>,
    /// Control points the final pass measured, for the run log.
    pub(crate) evaluated_control_points: usize,
}

/// Measure the residual of the exact inverse map that final RAW sampling uses.
/// Depth-band geometry must already be present in the observations; otherwise
/// applying the fitted residual on top of that geometry corrects it twice.
#[allow(clippy::too_many_arguments)]
pub(crate) fn register_station_frame_native_with_mapping(
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
    native_mapping: Option<&(dyn Fn(Point3<f64>) -> Option<Point2<f64>> + Sync)>,
) -> Option<StationFrameRegistration> {
    let inverse = transform.try_inverse()?;
    let (left, right, top, bottom) = layer;
    if right <= left || bottom <= top {
        return None;
    }
    let layer_width = right - left + 1;
    let layer_height = bottom - top + 1;
    // 需求 2.2: the two global + local rounds run at an analysis resolution whose
    // long side stays inside INTRA_STATION_ANALYSIS_LONG_SIDE; the native patch
    // refinement then runs on the native samples this sampler reaches.
    let scale = (layer_width.max(layer_height) as f64 / ANALYSIS_LONG_SIDE as f64).max(1.0);
    let analysis_width = (layer_width as f64 / scale).ceil() as u32;
    let analysis_height = (layer_height as f64 / scale).ceil() as u32;
    let mut sampler = LayerSampler {
        info,
        source,
        source_divisor: 1.0,
        inverse,
        native_mapping,
        projection,
        offset,
        left,
        top,
        scale,
        residual: Field::new(analysis_width, analysis_height, GRID_STEP as f64),
    };
    for _ in 0..intra_station::INTRA_STATION_ANALYSIS_ROUNDS {
        refine_layer(
            base,
            base_mask,
            &mut sampler,
            analysis_width,
            analysis_height,
        );
    }
    let outcome = refine_native_layer(
        base,
        base_mask,
        &mut sampler,
        analysis_width,
        analysis_height,
        NativeRefineMode::IntraStation { anchor_long_side },
    );
    let record = intra_station_frame_record(&info.filename, into_anchor, &outcome);
    Some(StationFrameRegistration {
        record,
        displacement: (!outcome.skipped).then(|| NativeResidualDisplacement {
            field: sampler.residual,
            left,
            top,
            scale,
        }),
        evaluated_control_points: outcome.evaluated_control_points,
    })
}
