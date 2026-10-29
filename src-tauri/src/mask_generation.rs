#[cfg(test)]
use base64::{Engine as _, engine::general_purpose};
use image::{DynamicImage, GenericImageView, ImageFormat, Rgba, RgbaImage};
#[cfg(test)]
use image::{GrayImage, Luma};
use serde_json::Value;
use std::borrow::Cow;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Cursor;
use std::sync::Arc; // Required for parallel rasterization
use tauri::ipc::Response;

use crate::app_state::{AppState, SharedMaskBitmap};
use crate::image_processing::{
    GeometryWarpSampler, apply_cpu_default_raw_processing, get_geometry_params_from_json,
};
#[cfg(test)]
use crate::render_strategy::GPU_TILE_SIZE;

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(crate = "serde")]
#[serde(rename_all = "camelCase")]
pub enum SubMaskMode {
    Additive,
    Subtractive,
    Intersect,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(crate = "serde")]
#[serde(rename_all = "camelCase")]
pub struct SubMask {
    pub id: String,
    #[serde(rename = "type")]
    pub mask_type: String,
    pub visible: bool,
    #[serde(default)]
    pub invert: bool,
    #[serde(default = "default_opacity")]
    pub opacity: f32,
    pub mode: SubMaskMode,
    pub parameters: Value,
}

fn default_opacity() -> f32 {
    100.0
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(crate = "serde")]
#[serde(rename_all = "camelCase")]
pub struct MaskDefinition {
    pub id: String,
    pub name: String,
    pub visible: bool,
    pub invert: bool,
    #[serde(default = "default_opacity")]
    pub opacity: f32,
    pub adjustments: Value,
    pub sub_masks: Vec<SubMask>,
}

impl MaskDefinition {
    pub fn requires_range_source(&self) -> bool {
        self.sub_masks
            .iter()
            .any(|sm| sm.mask_type == "color" || sm.mask_type == "luminance")
    }
}

pub trait RangeMaskSource: Sync {
    fn source_dimensions(&self) -> (u32, u32);
    fn sample_rgba8_points(&self, points: &[(u32, u32)], output: &mut [Rgba<u8>]) -> bool;

    fn sample_rgba8(&self, x: u32, y: u32) -> Option<Rgba<u8>> {
        let mut output = [Rgba([0, 0, 0, 0])];
        self.sample_rgba8_points(&[(x, y)], &mut output)
            .then_some(output[0])
    }
}

impl RangeMaskSource for DynamicImage {
    fn source_dimensions(&self) -> (u32, u32) {
        GenericImageView::dimensions(self)
    }

    fn sample_rgba8_points(&self, points: &[(u32, u32)], output: &mut [Rgba<u8>]) -> bool {
        let (width, height) = GenericImageView::dimensions(self);
        if points.len() != output.len() || points.iter().any(|(x, y)| *x >= width || *y >= height) {
            return false;
        }
        for ((x, y), output) in points.iter().copied().zip(output) {
            *output = GenericImageView::get_pixel(self, x, y);
        }
        true
    }
}

impl RangeMaskSource for GeometryWarpSampler<'_> {
    fn source_dimensions(&self) -> (u32, u32) {
        GeometryWarpSampler::dimensions(self)
    }

    fn sample_rgba8_points(&self, points: &[(u32, u32)], output: &mut [Rgba<u8>]) -> bool {
        GeometryWarpSampler::sample_rgba8_points(self, points, output)
    }
}

pub fn prepare_range_mask_source_image<'a>(
    image: &'a DynamicImage,
    is_raw: bool,
    masks: &[MaskDefinition],
) -> Option<Cow<'a, DynamicImage>> {
    if !masks.iter().any(MaskDefinition::requires_range_source) {
        return None;
    }
    if !is_raw {
        return Some(Cow::Borrowed(image));
    }

    let mut display_image = image.clone();
    apply_cpu_default_raw_processing(&mut display_image);
    Some(Cow::Owned(display_image))
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(crate = "serde")]
#[serde(rename_all = "camelCase")]
pub struct PatchData {
    pub color: String,
    pub mask: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(crate = "serde")]
#[serde(rename_all = "camelCase")]
pub struct AiPatchDefinition {
    pub id: String,
    pub name: String,
    pub visible: bool,
    pub invert: bool,
    pub prompt: String,
    #[serde(default)]
    pub patch_data: Option<PatchData>,
    #[serde(default = "default_opacity")]
    pub opacity: f32,
    pub sub_masks: Vec<SubMask>,
}

mod parameters;

use parameters::{
    BrushMaskParameters, FlowMaskParameters, GrowFeatherParameters, LinearMaskParameters,
    ParametricMaskParameters, Point, RadialMaskParameters,
};

mod grow_feather;

use grow_feather::{GrowFeatherPlan, apply_grow_and_feather};

mod shape_masks;

use shape_masks::{
    generate_brush_bitmap, generate_flow_bitmap, generate_linear_bitmap, generate_radial_bitmap,
};

mod ai_masks;

#[cfg(test)]
use ai_masks::{AiDepthSelection, AiMaskTransform, image_gaussian_blur_radius};
use ai_masks::{
    TiledAiMaskRasterizer, generate_ai_depth_bitmap, generate_ai_foreground_bitmap,
    generate_ai_sky_bitmap, generate_ai_subject_bitmap,
};

mod range_masks;

use range_masks::{
    generate_color_bitmap, generate_color_bitmap_from_parameters, generate_luminance_bitmap,
    generate_luminance_bitmap_from_parameters,
};

mod composition;

pub use composition::generate_mask_bitmap;
#[cfg(test)]
use composition::{
    apply_sub_mask_modifiers, composite_sub_mask, composite_sub_mask_tile,
    composite_sub_mask_tile_region, composite_sub_mask_tiled_with_edge, generate_sub_mask_bitmap,
    should_generate_sub_mask_in_tiles,
};

#[tauri::command]
pub fn generate_mask_overlay(
    mut mask_def: serde_json::Value,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    mut js_adjustments: Option<serde_json::Value>,
    state: tauri::State<'_, AppState>,
) -> Result<Response, String> {
    if let Some(ref mut adj) = js_adjustments {
        crate::adjustment_utils::hydrate_adjustments(&state, adj);
    }

    if let Some(sub_masks) = mask_def.get_mut("subMasks").and_then(|v| v.as_array_mut()) {
        let mut cache = state.patch_cache.lock().unwrap();
        crate::adjustment_utils::hydrate_sub_masks(sub_masks, &mut cache);
    }

    let parsed_mask_def: MaskDefinition = serde_json::from_value(mask_def)
        .map_err(|e| format!("Failed to parse hydrated mask_def: {}", e))?;

    let scaled_crop_offset = (crop_offset.0 * scale, crop_offset.1 * scale);

    let range_source_image = js_adjustments.as_ref().and_then(|_| {
        resolve_range_mask_source_image(&state, std::slice::from_ref(&parsed_mask_def))
    });
    let range_source = js_adjustments
        .as_ref()
        .zip(range_source_image.as_deref())
        .map(|(adjustments, image)| {
            GeometryWarpSampler::new(image, get_geometry_params_from_json(adjustments))
        });

    if let Some(gray_mask) = generate_mask_bitmap(
        &parsed_mask_def,
        width,
        height,
        scale,
        scaled_crop_offset,
        range_source
            .as_ref()
            .map(|source| source as &dyn RangeMaskSource),
    ) {
        let mut rgba_mask = RgbaImage::new(width, height);
        for (x, y, pixel) in gray_mask.enumerate_pixels() {
            let intensity = pixel[0];
            let alpha = (intensity as f32 * 0.5) as u8;
            rgba_mask.put_pixel(x, y, Rgba([255, 0, 0, alpha]));
        }

        let mut buf = Cursor::new(Vec::new());
        rgba_mask
            .write_to(&mut buf, ImageFormat::Png)
            .map_err(|e| e.to_string())?;

        Ok(Response::new(buf.into_inner()))
    } else {
        Ok(Response::new(Vec::new()))
    }
}

pub fn resolve_range_mask_source_image(
    state: &tauri::State<AppState>,
    masks: &[MaskDefinition],
) -> Option<Arc<DynamicImage>> {
    if !masks.iter().any(MaskDefinition::requires_range_source) {
        return None;
    }

    let (base_image, is_raw) = crate::get_original_image(state).ok()?;
    if !is_raw {
        return Some(base_image);
    }

    if let Ok(mut full_warped_cache) = state.full_warped_cache.lock() {
        *full_warped_cache = None;
    }
    let mut cache = state.range_mask_source_cache.lock().ok()?;
    if let Some((cached_base, cached_source)) = cache.as_ref()
        && Arc::ptr_eq(cached_base, &base_image)
    {
        return Some(Arc::clone(cached_source));
    }

    let mut display_image = base_image.as_ref().clone();
    apply_cpu_default_raw_processing(&mut display_image);
    let display_image = Arc::new(display_image);
    *cache = Some((base_image, Arc::clone(&display_image)));
    Some(display_image)
}

pub fn get_cached_or_generate_mask(
    state: &tauri::State<AppState>,
    def: &MaskDefinition,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    adjustments: &serde_json::Value,
) -> Option<SharedMaskBitmap> {
    let mut hasher = DefaultHasher::new();

    let mut def_for_hash = def.clone();
    def_for_hash.adjustments = serde_json::Value::Null;
    let def_json = serde_json::to_string(&def_for_hash).unwrap_or_default();
    def_json.hash(&mut hasher);

    width.hash(&mut hasher);
    height.hash(&mut hasher);
    scale.to_bits().hash(&mut hasher);
    crop_offset.0.to_bits().hash(&mut hasher);
    crop_offset.1.to_bits().hash(&mut hasher);
    if def.requires_range_source() {
        crate::cache_utils::calculate_geometry_hash(adjustments).hash(&mut hasher);
    }

    let key = hasher.finish();

    {
        let mut cache = state.mask_cache.lock().unwrap();
        if let Some(img) = cache.get(&key) {
            return Some(Arc::clone(img));
        }
    }

    let range_source_image = resolve_range_mask_source_image(state, std::slice::from_ref(def));
    let range_source = range_source_image
        .as_deref()
        .map(|image| GeometryWarpSampler::new(image, get_geometry_params_from_json(adjustments)));

    let generated = generate_mask_bitmap(
        def,
        width,
        height,
        scale,
        crop_offset,
        range_source
            .as_ref()
            .map(|source| source as &dyn RangeMaskSource),
    )
    .map(Arc::new);

    if let Some(img) = &generated {
        let mut cache = state.mask_cache.lock().unwrap();
        cache.insert(key, Arc::clone(img), img.as_raw().len());
    }

    generated
}

#[cfg(test)]
mod tests;
