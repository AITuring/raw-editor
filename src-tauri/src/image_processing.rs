use crate::gpu_processing::WgpuDisplay;
use image::DynamicImage;
use std::borrow::Cow;
use std::sync::Arc;

use crate::AppState;
pub use crate::gpu_processing::{
    RenderRequest, get_or_init_gpu_context, process_and_get_dynamic_image,
    process_and_get_dynamic_image_with_analytics,
};
#[cfg(test)]
use half::f16;
#[cfg(test)]
use image::GenericImageView;
#[cfg(test)]
use image::Rgb32FImage;
#[cfg(test)]
use image::RgbImage;
#[cfg(test)]
use serde_json::Value;

pub trait IntoCowImage<'a> {
    fn into_cow(self) -> Cow<'a, DynamicImage>;
}

impl<'a> IntoCowImage<'a> for DynamicImage {
    fn into_cow(self) -> Cow<'a, DynamicImage> {
        Cow::Owned(self)
    }
}

impl<'a> IntoCowImage<'a> for &'a DynamicImage {
    fn into_cow(self) -> Cow<'a, DynamicImage> {
        Cow::Borrowed(self)
    }
}

impl<'a> IntoCowImage<'a> for Cow<'a, DynamicImage> {
    fn into_cow(self) -> Cow<'a, DynamicImage> {
        self
    }
}

impl<'a> IntoCowImage<'a> for &'a std::sync::Arc<DynamicImage> {
    fn into_cow(self) -> Cow<'a, DynamicImage> {
        Cow::Borrowed(self.as_ref())
    }
}

#[derive(Clone)]
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub limits: wgpu::Limits,
    pub display: Arc<std::sync::Mutex<Option<WgpuDisplay>>>,
}

mod sidecar;

pub use sidecar::{ImageMetadata, SIDECAR_SCHEMA_VERSION, is_image_edited};

mod geometry_params;

pub use geometry_params::{
    Crop, GeometryParams, get_geometry_params_from_json, is_geometry_identity,
};

mod downscale;

pub use downscale::downscale_f32_image;

mod geometry_sampling;

#[cfg(test)]
use geometry_sampling::GeometrySampleData;
use geometry_sampling::{
    GeometryPixelSource, GeometrySampleContext, GeometryWarpBufferPlan, interpolate_geometry_pixel,
    interpolate_pixel, interpolate_pixel_with_tca,
};

mod geometry_projection;

use geometry_projection::{
    build_transform_matrices, project_geometry_point, solve_generic_distortion_inv,
    unproject_geometry_point,
};

mod geometry_warp;

use geometry_warp::GeometryWarpContext;
pub use geometry_warp::{GeometryWarpSampler, warp_image_geometry};

mod geometry_warp_rows;

#[cfg(test)]
use geometry_warp_rows::GeometryOrthogonalTransform;
pub use geometry_warp_rows::GeometryWarpRows;
use geometry_warp_rows::resolve_geometry_crop_rect;

mod geometry_ops;

pub use geometry_ops::{
    apply_coarse_rotation, apply_crop, apply_flip, apply_geometry_warp, apply_orientation,
    apply_rotation, inverse_transform_mask, inverse_transform_point,
};

mod color_transforms;

use color_transforms::calculate_agx_matrices;
pub use color_transforms::{
    apply_cpu_agx_tonemap, apply_cpu_default_raw_processing, apply_linear_to_srgb,
    apply_srgb_to_linear, resolve_tonemapper_override, resolve_tonemapper_override_from_handle,
};

mod adjustments;

pub use adjustments::{
    AllAdjustments, ColorCalibrationSettings, ColorGradeSettings, GlobalAdjustments, GpuMat3,
    HslColor, MAX_MASKS, MaskAdjustments, Point,
};

mod adjustment_parsing;

pub use adjustment_parsing::get_all_adjustments_from_json;

mod raw_enhance;

pub use raw_enhance::remove_raw_artifacts_and_enhance;

mod scopes;

pub use scopes::{calculate_histogram_from_image, calculate_waveform_from_image};

mod auto_adjust;

pub use auto_adjust::{auto_results_to_json, perform_auto_analysis};
#[cfg(test)]
use auto_adjust::{estimate_auto_white_balance, white_balance_multipliers};

#[cfg(test)]
mod raw_enhancement_tests;

#[tauri::command]
pub fn calculate_auto_adjustments(
    state: tauri::State<AppState>,
) -> Result<serde_json::Value, String> {
    let original_image = state
        .original_image
        .lock()
        .unwrap()
        .as_ref()
        .ok_or("No image loaded for auto adjustments")?
        .image
        .clone();

    let results = perform_auto_analysis(&original_image);

    Ok(auto_results_to_json(&results))
}

#[cfg(test)]
mod geometry_transform_tests;

#[cfg(test)]
mod color_transfer_tests;

#[cfg(test)]
mod auto_white_balance_tests;
