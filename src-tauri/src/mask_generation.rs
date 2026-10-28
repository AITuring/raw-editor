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
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    use super::*;
    use crate::cache_utils::BudgetedCache;

    #[test]
    fn first_additive_submask_reuses_its_pixel_allocation() {
        let additive = GrayImage::from_fn(17, 11, |x, y| Luma([((x * 13 + y * 7) % 256) as u8]));
        let additive_ptr = additive.as_raw().as_ptr();
        let expected = additive.clone();
        let mut composed = None;

        composite_sub_mask(&mut composed, additive, SubMaskMode::Additive);

        let composed = composed.expect("additive bitmap becomes the composition base");
        assert_eq!(composed.as_raw().as_ptr(), additive_ptr);
        assert_eq!(composed, expected);
    }

    #[test]
    fn lazy_mask_composition_preserves_black_base_semantics() {
        let mut composed = None;
        composite_sub_mask(
            &mut composed,
            GrayImage::from_pixel(5, 3, Luma([200])),
            SubMaskMode::Subtractive,
        );
        composite_sub_mask(
            &mut composed,
            GrayImage::from_pixel(5, 3, Luma([100])),
            SubMaskMode::Intersect,
        );
        assert!(composed.is_none());

        composite_sub_mask(
            &mut composed,
            GrayImage::from_pixel(5, 3, Luma([180])),
            SubMaskMode::Additive,
        );
        composite_sub_mask(
            &mut composed,
            GrayImage::from_pixel(5, 3, Luma([30])),
            SubMaskMode::Subtractive,
        );
        composite_sub_mask(
            &mut composed,
            GrayImage::from_pixel(5, 3, Luma([120])),
            SubMaskMode::Intersect,
        );

        assert!(
            composed
                .expect("additive submask establishes the output")
                .pixels()
                .all(|pixel| pixel[0] == 120)
        );
    }

    #[test]
    fn shared_mask_cache_keeps_one_60mp_pixel_allocation() {
        const WIDTH: usize = 9_504;
        const HEIGHT: usize = 6_336;
        const MASK_BYTES: usize = 60_217_344;

        assert_eq!(WIDTH * HEIGHT, MASK_BYTES);
        let mask = Arc::new(GrayImage::from_pixel(7, 5, Luma([123])));
        let mut cache = BudgetedCache::new(2, 1_024);
        assert!(cache.insert(42_u64, Arc::clone(&mask), mask.as_raw().len()));
        let caller = Arc::clone(cache.get(&42).expect("shared mask cache hit"));

        assert!(Arc::ptr_eq(&mask, &caller));
        assert_eq!(mask.as_raw().as_ptr(), caller.as_raw().as_ptr());
        assert_eq!(MASK_BYTES * 2 - MASK_BYTES, MASK_BYTES);
    }

    #[test]
    fn range_mask_source_preparation_borrows_non_raw_and_preserves_raw_display_semantics() {
        let source = DynamicImage::ImageRgb32F(image::Rgb32FImage::from_fn(7, 5, |x, y| {
            image::Rgb([
                (x + 1) as f32 / 8.0,
                (y + 1) as f32 / 6.0,
                (x + y + 1) as f32 / 12.0,
            ])
        }));
        let original = source.clone();
        let range_mask = MaskDefinition {
            id: "range-source".to_string(),
            name: "Range source".to_string(),
            visible: true,
            invert: false,
            opacity: 100.0,
            adjustments: Value::Null,
            sub_masks: vec![test_sub_mask(
                "color",
                serde_json::json!({}),
                SubMaskMode::Additive,
            )],
        };

        assert!(prepare_range_mask_source_image(&source, false, &[]).is_none());
        let borrowed =
            prepare_range_mask_source_image(&source, false, std::slice::from_ref(&range_mask))
                .expect("non-RAW range masks need a source");
        assert!(matches!(borrowed, Cow::Borrowed(image) if std::ptr::eq(image, &source)));

        let mut expected = source.clone();
        apply_cpu_default_raw_processing(&mut expected);
        let processed =
            prepare_range_mask_source_image(&source, true, std::slice::from_ref(&range_mask))
                .expect("RAW range masks need a display source")
                .into_owned();
        assert_eq!(processed, expected);
        assert_eq!(
            source, original,
            "RAW preparation must not mutate the source"
        );
    }

    fn test_sub_mask(mask_type: &str, parameters: Value, mode: SubMaskMode) -> SubMask {
        SubMask {
            id: format!("test-{mask_type}"),
            mask_type: mask_type.to_string(),
            visible: true,
            invert: true,
            opacity: 63.0,
            mode,
            parameters,
        }
    }

    fn grayscale_png_data_url(image: &GrayImage) -> String {
        let mut encoded = Cursor::new(Vec::new());
        image
            .write_to(&mut encoded, ImageFormat::Png)
            .expect("encode deterministic AI mask fixture");
        format!(
            "data:image/png;base64,{}",
            general_purpose::STANDARD.encode(encoded.get_ref())
        )
    }

    fn brush_parameters(flow: bool) -> Value {
        let first_line = if flow {
            serde_json::json!({
                "tool": "brush",
                "brushSize": 29.0,
                "feather": 0.37,
                "flow": 42.0,
                "points": [{ "x": 8.0, "y": 12.0 }, { "x": 128.0, "y": 91.0 }]
            })
        } else {
            serde_json::json!({
                "tool": "brush",
                "brushSize": 29.0,
                "feather": 0.37,
                "points": [{ "x": 8.0, "y": 12.0 }, { "x": 128.0, "y": 91.0 }]
            })
        };
        let eraser_line = if flow {
            serde_json::json!({
                "tool": "eraser",
                "brushSize": 17.0,
                "feather": 0.61,
                "flow": 73.0,
                "points": [{ "x": 76.0, "y": 5.0 }, { "x": 65.0, "y": 104.0 }]
            })
        } else {
            serde_json::json!({
                "tool": "eraser",
                "brushSize": 17.0,
                "feather": 0.61,
                "points": [{ "x": 76.0, "y": 5.0 }, { "x": 65.0, "y": 104.0 }]
            })
        };
        serde_json::json!({ "lines": [first_line, eraser_line] })
    }

    #[test]
    fn tiled_programmatic_and_brush_composition_matches_full_frame_across_seams() {
        const WIDTH: u32 = 141;
        const HEIGHT: u32 = 109;
        const TEST_TILE_EDGE: u32 = 64;
        const SCALE: f32 = 1.25;
        const CROP_OFFSET: (f32, f32) = (11.5, 7.25);

        let cases = [
            (
                "radial",
                serde_json::json!({
                    "centerX": 61.0,
                    "centerY": 47.0,
                    "radiusX": 54.0,
                    "radiusY": 31.0,
                    "rotation": 23.0,
                    "feather": 0.41
                }),
            ),
            (
                "linear",
                serde_json::json!({
                    "startX": 17.0,
                    "startY": 19.0,
                    "endX": 119.0,
                    "endY": 88.0,
                    "range": 38.0
                }),
            ),
            ("brush", brush_parameters(false)),
            ("clone", brush_parameters(false)),
            ("heal", brush_parameters(false)),
            ("flow", brush_parameters(true)),
            ("all", serde_json::json!({})),
        ];
        let modes = [
            SubMaskMode::Additive,
            SubMaskMode::Subtractive,
            SubMaskMode::Intersect,
        ];

        for (mask_type, parameters) in cases {
            for mode in modes {
                let sub_mask = test_sub_mask(mask_type, parameters.clone(), mode);
                let base = GrayImage::from_fn(WIDTH, HEIGHT, |x, y| {
                    Luma([x.wrapping_mul(11).wrapping_add(y * 7) as u8])
                });

                let mut expected = (mode != SubMaskMode::Additive).then(|| base.clone());
                let mut full_sub_mask =
                    generate_sub_mask_bitmap(&sub_mask, WIDTH, HEIGHT, SCALE, CROP_OFFSET, None)
                        .expect("test mask type should rasterize");
                apply_sub_mask_modifiers(&mut full_sub_mask, &sub_mask);
                composite_sub_mask(&mut expected, full_sub_mask, mode);

                let mut actual = (mode != SubMaskMode::Additive).then_some(base);
                composite_sub_mask_tiled_with_edge(
                    &mut actual,
                    &sub_mask,
                    WIDTH,
                    HEIGHT,
                    SCALE,
                    CROP_OFFSET,
                    None,
                    TEST_TILE_EDGE,
                );

                let actual = actual.expect("tiled composition output");
                let expected = expected.expect("full-frame composition output");
                let first_difference = actual
                    .as_raw()
                    .iter()
                    .zip(expected.as_raw())
                    .position(|(actual, expected)| actual != expected)
                    .map(|index| {
                        (
                            index as u32 % WIDTH,
                            index as u32 / WIDTH,
                            actual.as_raw()[index],
                            expected.as_raw()[index],
                        )
                    });
                assert_eq!(
                    first_difference, None,
                    "{mask_type} {mode:?} changed at a tile boundary"
                );
            }
        }
    }

    #[test]
    fn first_brush_and_flow_submasks_match_full_frame_when_generated_in_tiles() {
        const WIDTH: u32 = 141;
        const HEIGHT: u32 = 109;

        for (mask_type, parameters) in [
            ("brush", brush_parameters(false)),
            ("flow", brush_parameters(true)),
        ] {
            let sub_mask = test_sub_mask(mask_type, parameters, SubMaskMode::Additive);
            let mut expected_sub_mask =
                generate_sub_mask_bitmap(&sub_mask, WIDTH, HEIGHT, 1.0, (0.0, 0.0), None)
                    .expect("brush-like test mask should rasterize");
            apply_sub_mask_modifiers(&mut expected_sub_mask, &sub_mask);

            let mut actual = None;
            composite_sub_mask_tiled_with_edge(
                &mut actual,
                &sub_mask,
                WIDTH,
                HEIGHT,
                1.0,
                (0.0, 0.0),
                None,
                64,
            );

            assert_eq!(actual, Some(expected_sub_mask), "{mask_type}");
        }
    }

    #[test]
    fn tiled_geometry_sampled_color_and_luminance_masks_match_full_frame_with_exact_halo() {
        const WIDTH: u32 = 141;
        const HEIGHT: u32 = 109;
        const TEST_TILE_EDGE: u32 = 37;
        const SCALE: f32 = 1.25;
        const CROP_OFFSET: (f32, f32) = (11.5, 7.25);

        let source_image = DynamicImage::ImageRgba8(RgbaImage::from_fn(181, 137, |x, y| {
            Rgba([
                x.wrapping_mul(17).wrapping_add(y * 3) as u8,
                x.wrapping_mul(5).wrapping_add(y * 19) as u8,
                x.wrapping_mul(11).wrapping_add(y * 7) as u8,
                255,
            ])
        }));
        let geometry_params = crate::image_processing::GeometryParams {
            distortion: 8.0,
            projection: 14.0,
            vertical: 9.0,
            horizontal: -7.0,
            rotate: 2.0,
            aspect: 3.0,
            scale: 98.0,
            x_offset: 1.0,
            y_offset: -1.0,
            lens_dist_k1: 0.008,
            tca_vr: 1.001,
            tca_vb: 0.999,
            vig_k1: -0.04,
            ..crate::image_processing::GeometryParams::default()
        };
        let warped = crate::image_processing::warp_image_geometry(&source_image, geometry_params);
        let range_source = GeometryWarpSampler::new(&source_image, geometry_params);
        let cases = [
            (
                "color",
                serde_json::json!({
                    "targetX": 72.0,
                    "targetY": 58.0,
                    "tolerance": 72.0,
                    "grow": 100.0,
                    "feather": 100.0,
                    "rotation": 17.0,
                    "flipHorizontal": true,
                    "flipVertical": false,
                    "orientationSteps": 1
                }),
            ),
            (
                "luminance",
                serde_json::json!({
                    "targetX": 113.0,
                    "targetY": 41.0,
                    "tolerance": 32.0,
                    "grow": -100.0,
                    "feather": 100.0,
                    "rotation": -13.0,
                    "flipHorizontal": false,
                    "flipVertical": true,
                    "orientationSteps": 2
                }),
            ),
        ];

        for (mask_type, parameters) in cases {
            for mode in [
                SubMaskMode::Additive,
                SubMaskMode::Subtractive,
                SubMaskMode::Intersect,
            ] {
                let sub_mask = test_sub_mask(mask_type, parameters.clone(), mode);
                let base = GrayImage::from_fn(WIDTH, HEIGHT, |x, y| {
                    Luma([x.wrapping_mul(11).wrapping_add(y * 7) as u8])
                });

                let mut expected = (mode != SubMaskMode::Additive).then(|| base.clone());
                let mut full_sub_mask = generate_sub_mask_bitmap(
                    &sub_mask,
                    WIDTH,
                    HEIGHT,
                    SCALE,
                    CROP_OFFSET,
                    Some(&warped),
                )
                .expect("range mask should rasterize");
                let first_pixel = full_sub_mask.get_pixel(0, 0)[0];
                assert!(
                    full_sub_mask.pixels().any(|pixel| pixel[0] != first_pixel),
                    "{mask_type} fixture must exercise non-uniform pixels"
                );
                apply_sub_mask_modifiers(&mut full_sub_mask, &sub_mask);
                composite_sub_mask(&mut expected, full_sub_mask, mode);

                let mut actual = (mode != SubMaskMode::Additive).then_some(base);
                assert!(composite_sub_mask_tiled_with_edge(
                    &mut actual,
                    &sub_mask,
                    WIDTH,
                    HEIGHT,
                    SCALE,
                    CROP_OFFSET,
                    Some(&range_source),
                    TEST_TILE_EDGE,
                ));

                let actual = actual.expect("tiled composition output");
                let expected = expected.expect("full-frame composition output");
                let first_difference = actual
                    .as_raw()
                    .iter()
                    .zip(expected.as_raw())
                    .position(|(actual, expected)| actual != expected)
                    .map(|index| {
                        (
                            index as u32 % WIDTH,
                            index as u32 / WIDTH,
                            actual.as_raw()[index],
                            expected.as_raw()[index],
                        )
                    });
                assert_eq!(
                    first_difference, None,
                    "{mask_type} {mode:?} changed at an overlap seam"
                );
            }
        }
    }

    #[test]
    fn tiled_ai_masks_match_full_frame_with_exact_halo_and_one_decoded_source() {
        const WIDTH: u32 = 141;
        const HEIGHT: u32 = 109;
        const TEST_TILE_EDGE: u32 = 37;
        const SCALE: f32 = 1.25;
        const CROP_OFFSET: (f32, f32) = (11.5, 7.25);

        let full_mask = GrayImage::from_fn(181, 137, |x, y| {
            Luma([x.wrapping_mul(17).wrapping_add(y.wrapping_mul(13)) as u8])
        });
        let data_url = grayscale_png_data_url(&full_mask);
        let binary_parameters = serde_json::json!({
            "startX": 12.0,
            "startY": 9.0,
            "endX": 128.0,
            "endY": 96.0,
            "maskDataBase64": data_url,
            "grow": 100.0,
            "feather": 100.0,
            "rotation": 17.0,
            "flipHorizontal": true,
            "flipVertical": false,
            "orientationSteps": 1
        });
        let cases = vec![
            ("ai-subject", binary_parameters.clone()),
            ("ai-foreground", binary_parameters.clone()),
            ("ai-sky", binary_parameters.clone()),
            ("quick-eraser", binary_parameters),
            (
                "ai-depth",
                serde_json::json!({
                    "maskDataBase64": grayscale_png_data_url(&full_mask),
                    "minDepth": 18.0,
                    "maxDepth": 82.0,
                    "minFade": 11.0,
                    "maxFade": 9.0,
                    "grow": -100.0,
                    "feather": 60.0,
                    "rotation": -13.0,
                    "flipHorizontal": false,
                    "flipVertical": true,
                    "orientationSteps": 2
                }),
            ),
        ];

        for (mask_type, parameters) in cases {
            for mode in [
                SubMaskMode::Additive,
                SubMaskMode::Subtractive,
                SubMaskMode::Intersect,
            ] {
                let sub_mask = test_sub_mask(mask_type, parameters.clone(), mode);
                let base = GrayImage::from_fn(WIDTH, HEIGHT, |x, y| {
                    Luma([x.wrapping_mul(11).wrapping_add(y * 7) as u8])
                });

                let mut expected = (mode != SubMaskMode::Additive).then(|| base.clone());
                let mut full_sub_mask =
                    generate_sub_mask_bitmap(&sub_mask, WIDTH, HEIGHT, SCALE, CROP_OFFSET, None)
                        .expect("AI mask should decode and rasterize");
                let first_pixel = full_sub_mask.get_pixel(0, 0)[0];
                assert!(
                    full_sub_mask.pixels().any(|pixel| pixel[0] != first_pixel),
                    "{mask_type} fixture must exercise non-uniform pixels"
                );
                apply_sub_mask_modifiers(&mut full_sub_mask, &sub_mask);
                composite_sub_mask(&mut expected, full_sub_mask, mode);

                let mut actual = (mode != SubMaskMode::Additive).then_some(base);
                assert!(composite_sub_mask_tiled_with_edge(
                    &mut actual,
                    &sub_mask,
                    WIDTH,
                    HEIGHT,
                    SCALE,
                    CROP_OFFSET,
                    None,
                    TEST_TILE_EDGE,
                ));

                let actual = actual.expect("tiled AI composition output");
                let expected = expected.expect("full-frame AI composition output");
                let first_difference = actual
                    .as_raw()
                    .iter()
                    .zip(expected.as_raw())
                    .position(|(actual, expected)| actual != expected)
                    .map(|index| {
                        (
                            index as u32 % WIDTH,
                            index as u32 / WIDTH,
                            actual.as_raw()[index],
                            expected.as_raw()[index],
                        )
                    });
                assert_eq!(
                    first_difference, None,
                    "{mask_type} {mode:?} changed at an overlap seam"
                );
            }
        }
    }

    #[test]
    fn ai_depth_halo_bounds_supported_60mp_filter_scratch() {
        const WIDTH: u32 = 9_504;
        const HEIGHT: u32 = 6_336;
        const FULL_MASK_BYTES: u64 = 60_217_344;
        const MAX_AI_DEPTH_HALO: u64 = 159;
        const MAX_EXPANDED_TILE_EDGE: u64 = 2_366;
        const MAX_EXPANDED_TILE_BYTES: u64 = 5_597_956;

        assert_eq!(image_gaussian_blur_radius(0.0), 0);
        assert_eq!(image_gaussian_blur_radius(0.1), 1);
        assert_eq!(image_gaussian_blur_radius(1.5), 3);
        assert_eq!(image_gaussian_blur_radius(10.0), 32);

        let grow_feather = GrowFeatherPlan::new(100.0, 100.0, WIDTH, HEIGHT);
        assert_eq!(
            u64::from(grow_feather.halo()) + u64::from(image_gaussian_blur_radius(10.0)),
            MAX_AI_DEPTH_HALO
        );
        assert_eq!(
            u64::from(GPU_TILE_SIZE) + MAX_AI_DEPTH_HALO * 2,
            MAX_EXPANDED_TILE_EDGE
        );
        assert_eq!(MAX_EXPANDED_TILE_EDGE.pow(2), MAX_EXPANDED_TILE_BYTES);
        assert_eq!(u64::from(WIDTH) * u64::from(HEIGHT), FULL_MASK_BYTES);
        assert_eq!(
            FULL_MASK_BYTES * 3 - MAX_EXPANDED_TILE_BYTES * 3,
            163_858_164
        );
    }

    #[test]
    fn grow_feather_halo_bounds_supported_60mp_range_mask_scratch() {
        const WIDTH: u32 = 9_504;
        const HEIGHT: u32 = 6_336;
        const FULL_MASK_BYTES: u64 = 60_217_344;
        const MAX_EXPANDED_TILE_EDGE: u64 = 2_302;
        const MAX_EXPANDED_TILE_BYTES: u64 = 5_299_204;

        let plan = GrowFeatherPlan::new(100.0, 100.0, WIDTH, HEIGHT);
        assert_eq!(plan.grow_amount, 63);
        assert!(plan.dilate);
        assert!((plan.feather_sigma - 31.68).abs() < 0.0001);
        assert_eq!(plan.feather_radius, 64);
        assert_eq!(plan.halo(), 127);
        assert_eq!(
            u64::from(GPU_TILE_SIZE) + u64::from(plan.halo()) * 2,
            MAX_EXPANDED_TILE_EDGE
        );
        assert_eq!(MAX_EXPANDED_TILE_EDGE.pow(2), MAX_EXPANDED_TILE_BYTES);
        assert_eq!(u64::from(WIDTH) * u64::from(HEIGHT), FULL_MASK_BYTES);
        assert_eq!(
            FULL_MASK_BYTES * 3 - MAX_EXPANDED_TILE_BYTES * 3,
            164_754_420
        );
    }

    #[test]
    fn tiled_generation_bounds_60mp_programmatic_and_brush_scratch() {
        const WIDTH: usize = 9_504;
        const HEIGHT: usize = 6_336;
        const FULL_MASK_BYTES: usize = 60_217_344;
        const MAX_TILE_BYTES: usize = 4_194_304;

        assert_eq!(WIDTH * HEIGHT, FULL_MASK_BYTES);
        assert_eq!(
            WIDTH.min(GPU_TILE_SIZE as usize) * HEIGHT.min(GPU_TILE_SIZE as usize),
            MAX_TILE_BYTES
        );
        assert_eq!(
            FULL_MASK_BYTES * 2 - (FULL_MASK_BYTES + MAX_TILE_BYTES),
            56_023_040
        );
        assert_eq!(
            FULL_MASK_BYTES * 3 - (FULL_MASK_BYTES + MAX_TILE_BYTES * 2),
            112_046_080
        );
    }

    #[test]
    fn tiled_generation_routes_supported_nonempty_submasks() {
        let radial = test_sub_mask("radial", serde_json::json!({}), SubMaskMode::Additive);
        assert!(!should_generate_sub_mask_in_tiles(&radial, false, 100, 80));
        assert!(should_generate_sub_mask_in_tiles(&radial, true, 100, 80));

        let brush = test_sub_mask("brush", brush_parameters(false), SubMaskMode::Additive);
        assert!(should_generate_sub_mask_in_tiles(&brush, false, 100, 80));

        let mut empty_brush = brush.clone();
        empty_brush.parameters = serde_json::json!({ "lines": [] });
        assert!(!should_generate_sub_mask_in_tiles(
            &empty_brush,
            false,
            100,
            80
        ));

        let mut hidden_brush = brush;
        hidden_brush.visible = false;
        assert!(!should_generate_sub_mask_in_tiles(
            &hidden_brush,
            true,
            100,
            80
        ));

        let color = test_sub_mask("color", serde_json::json!({}), SubMaskMode::Additive);
        assert!(!should_generate_sub_mask_in_tiles(&color, true, 100, 80));

        let mut color = test_sub_mask(
            "color",
            serde_json::json!({
                "targetX": 0.0,
                "targetY": 0.0,
                "grow": 0.0,
                "feather": 0.0
            }),
            SubMaskMode::Additive,
        );
        assert!(!should_generate_sub_mask_in_tiles(
            &color, false, 9_504, 6_336
        ));
        assert!(should_generate_sub_mask_in_tiles(
            &color, true, 9_504, 6_336
        ));
        color.parameters["feather"] = serde_json::json!(35.0);
        assert!(should_generate_sub_mask_in_tiles(
            &color, false, 9_504, 6_336
        ));

        let mut ai_subject = test_sub_mask(
            "ai-subject",
            serde_json::json!({ "grow": 0.0, "feather": 0.0 }),
            SubMaskMode::Additive,
        );
        assert!(!should_generate_sub_mask_in_tiles(
            &ai_subject,
            false,
            9_504,
            6_336
        ));
        assert!(should_generate_sub_mask_in_tiles(
            &ai_subject,
            true,
            9_504,
            6_336
        ));
        ai_subject.parameters["grow"] = serde_json::json!(50.0);
        assert!(should_generate_sub_mask_in_tiles(
            &ai_subject,
            false,
            9_504,
            6_336
        ));

        let ai_depth = test_sub_mask(
            "ai-depth",
            serde_json::json!({ "grow": 0.0, "feather": 15.0 }),
            SubMaskMode::Additive,
        );
        assert!(should_generate_sub_mask_in_tiles(
            &ai_depth, false, 9_504, 6_336
        ));
    }

    #[test]
    #[ignore = "manual deterministic 60MP range-mask overlap scratch benchmark"]
    fn synthetic_60mp_range_mask_overlap_scratch_harness() {
        let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(9_504_u32);
        let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(6_336_u32);
        let mode = std::env::var("RAW_EDITOR_RANGE_MASK_BENCH_MODE")
            .unwrap_or_else(|_| "tiled".to_string());
        assert!(matches!(mode.as_str(), "full" | "tiled"));

        let grow_feather = GrowFeatherPlan::new(2.0, 1.0, width, height);
        let halo = grow_feather.halo();
        let mut composition = GrayImage::from_pixel(width, height, Luma([17]));
        let sub_mask = SubMask {
            id: "synthetic-range-mask".to_string(),
            mask_type: "color".to_string(),
            visible: true,
            invert: false,
            opacity: 100.0,
            mode: SubMaskMode::Additive,
            parameters: Value::Null,
        };
        let pid = sysinfo::get_current_pid().expect("resolve benchmark process id");
        let mut baseline_system = sysinfo::System::new();
        baseline_system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        let baseline_rss = baseline_system
            .process(pid)
            .expect("read benchmark process after output allocation")
            .memory();

        let running = Arc::new(AtomicBool::new(true));
        let peak_rss = Arc::new(AtomicU64::new(baseline_rss));
        let sampler_running = Arc::clone(&running);
        let sampler_peak = Arc::clone(&peak_rss);
        let sampler = std::thread::spawn(move || {
            let mut system = sysinfo::System::new();
            while sampler_running.load(Ordering::Relaxed) {
                system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
                if let Some(process) = system.process(pid) {
                    sampler_peak.fetch_max(process.memory(), Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        });

        let pixel =
            |x: u32, y: u32| Luma([x.wrapping_mul(17).wrapping_add(y.wrapping_mul(13)) as u8]);
        let started = Instant::now();
        if mode == "full" {
            let mut full_sub_mask = GrayImage::from_fn(width, height, pixel);
            grow_feather.apply(&mut full_sub_mask);
            composite_sub_mask_tile(
                &mut composition,
                &full_sub_mask,
                0,
                0,
                SubMaskMode::Additive,
            );
        } else {
            for tile_y in (0..height).step_by(GPU_TILE_SIZE as usize) {
                let tile_height = (height - tile_y).min(GPU_TILE_SIZE);
                for tile_x in (0..width).step_by(GPU_TILE_SIZE as usize) {
                    let tile_width = (width - tile_x).min(GPU_TILE_SIZE);
                    let expanded_x = tile_x.saturating_sub(halo);
                    let expanded_y = tile_y.saturating_sub(halo);
                    let expanded_right = tile_x
                        .saturating_add(tile_width)
                        .saturating_add(halo)
                        .min(width);
                    let expanded_bottom = tile_y
                        .saturating_add(tile_height)
                        .saturating_add(halo)
                        .min(height);
                    let mut tile = GrayImage::from_fn(
                        expanded_right - expanded_x,
                        expanded_bottom - expanded_y,
                        |x, y| pixel(x + expanded_x, y + expanded_y),
                    );
                    grow_feather.apply(&mut tile);
                    composite_sub_mask_tile_region(
                        &mut composition,
                        &mut tile,
                        tile_x - expanded_x,
                        tile_y - expanded_y,
                        tile_width,
                        tile_height,
                        tile_x,
                        tile_y,
                        &sub_mask,
                    );
                }
            }
        }
        let elapsed = started.elapsed();
        std::thread::sleep(Duration::from_millis(10));
        running.store(false, Ordering::Relaxed);
        sampler.join().expect("join range-mask RSS sampler");

        let sample_stride = (composition.as_raw().len() / 4_096).max(1);
        let sample_hash = composition
            .as_raw()
            .iter()
            .step_by(sample_stride)
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, sample| {
                (hash ^ u64::from(*sample)).wrapping_mul(0x0000_0100_0000_01b3)
            });
        let peak_rss = peak_rss.load(Ordering::Relaxed);
        let expected_scratch_bytes = if mode == "full" {
            u64::from(width) * u64::from(height) * 3
        } else {
            let expanded_width = width.min(GPU_TILE_SIZE.saturating_add(halo * 2));
            let expanded_height = height.min(GPU_TILE_SIZE.saturating_add(halo * 2));
            u64::from(expanded_width) * u64::from(expanded_height) * 3
        };
        println!(
            "{{\"mode\":\"{}\",\"width\":{},\"height\":{},\"haloPixels\":{},\"elapsedMs\":{},\"baselineRssBytes\":{},\"peakRssBytes\":{},\"peakDeltaBytes\":{},\"expectedScratchBytes\":{},\"sampleHash\":\"{:016x}\"}}",
            mode,
            width,
            height,
            halo,
            elapsed.as_millis(),
            baseline_rss,
            peak_rss,
            peak_rss.saturating_sub(baseline_rss),
            expected_scratch_bytes,
            sample_hash,
        );
        std::hint::black_box(composition);
    }

    #[test]
    #[ignore = "manual deterministic 60MP range-mask geometry-source benchmark"]
    fn synthetic_60mp_range_mask_geometry_source_harness() {
        let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(9_504_u32);
        let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(6_336_u32);
        let mode = std::env::var("RAW_EDITOR_RANGE_SOURCE_BENCH_MODE")
            .unwrap_or_else(|_| "sampled".to_string());
        assert!(matches!(mode.as_str(), "full" | "sampled"));

        let source =
            DynamicImage::ImageRgb32F(image::Rgb32FImage::from_fn(width, height, |x, y| {
                image::Rgb([
                    (x.wrapping_mul(17).wrapping_add(y * 13) & 1023) as f32 / 1023.0,
                    (x.wrapping_mul(5).wrapping_add(y * 19) & 1023) as f32 / 1023.0,
                    (x.wrapping_mul(11).wrapping_add(y * 7) & 1023) as f32 / 1023.0,
                ])
            }));
        let geometry_params = crate::image_processing::GeometryParams {
            distortion: 5.0,
            projection: 8.0,
            vertical: 6.0,
            horizontal: -4.0,
            rotate: 1.0,
            aspect: 2.0,
            scale: 99.0,
            x_offset: 0.5,
            y_offset: -0.5,
            lens_dist_k1: 0.004,
            tca_vr: 1.0005,
            tca_vb: 0.9995,
            vig_k1: -0.02,
            ..crate::image_processing::GeometryParams::default()
        };
        let mask_parameters = ParametricMaskParameters {
            target_x: f64::from(width / 2),
            target_y: f64::from(height / 2),
            tolerance: 28.0,
            grow: 0.0,
            feather: 0.0,
            rotation: 0.0,
            flip_horizontal: false,
            flip_vertical: false,
            orientation_steps: 0,
        };

        let pid = sysinfo::get_current_pid().expect("resolve benchmark process id");
        let mut baseline_system = sysinfo::System::new();
        baseline_system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        let baseline_rss = baseline_system
            .process(pid)
            .expect("read benchmark process after source allocation")
            .memory();
        let running = Arc::new(AtomicBool::new(true));
        let peak_rss = Arc::new(AtomicU64::new(baseline_rss));
        let sampler_running = Arc::clone(&running);
        let sampler_peak = Arc::clone(&peak_rss);
        let rss_sampler = std::thread::spawn(move || {
            let mut system = sysinfo::System::new();
            while sampler_running.load(Ordering::Relaxed) {
                system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
                if let Some(process) = system.process(pid) {
                    sampler_peak.fetch_max(process.memory(), Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        });

        let started = Instant::now();
        let (mask, checkpoint_bytes, maximum_checkpoint_bytes) = if mode == "full" {
            let warped = crate::image_processing::warp_image_geometry(&source, geometry_params);
            let mask = generate_color_bitmap_from_parameters(
                &mask_parameters,
                width,
                height,
                1.0,
                (0.0, 0.0),
                (0, 0),
                &warped,
            )
            .expect("materialized range mask must rasterize");
            std::hint::black_box(warped);
            (mask, 0, 0)
        } else {
            let geometry_source = GeometryWarpSampler::new(&source, geometry_params);
            let mask = generate_color_bitmap_from_parameters(
                &mask_parameters,
                width,
                height,
                1.0,
                (0.0, 0.0),
                (0, 0),
                &geometry_source,
            )
            .expect("sampled range mask must rasterize");
            let checkpoint_bytes = geometry_source.checkpoint_bytes();
            let maximum_checkpoint_bytes = geometry_source.max_checkpoint_bytes();
            (mask, checkpoint_bytes, maximum_checkpoint_bytes)
        };
        let elapsed = started.elapsed();
        std::thread::sleep(Duration::from_millis(10));
        running.store(false, Ordering::Relaxed);
        rss_sampler.join().expect("join range-source RSS sampler");

        let sample_stride = (mask.as_raw().len() / 4_096).max(1);
        let sample_hash = mask
            .as_raw()
            .iter()
            .step_by(sample_stride)
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, sample| {
                (hash ^ u64::from(*sample)).wrapping_mul(0x0000_0100_0000_01b3)
            });
        let peak_rss = peak_rss.load(Ordering::Relaxed);
        let materialized_source_bytes = u64::from(width) * u64::from(height) * 4 * 4;
        println!(
            "{{\"mode\":\"{}\",\"width\":{},\"height\":{},\"elapsedMs\":{},\"baselineRssBytes\":{},\"peakRssBytes\":{},\"peakDeltaBytes\":{},\"materializedSourceBytes\":{},\"checkpointBytes\":{},\"maximumCheckpointBytes\":{},\"sampleHash\":\"{:016x}\"}}",
            mode,
            width,
            height,
            elapsed.as_millis(),
            baseline_rss,
            peak_rss,
            peak_rss.saturating_sub(baseline_rss),
            materialized_source_bytes,
            checkpoint_bytes,
            maximum_checkpoint_bytes,
            sample_hash,
        );
        std::hint::black_box(mask);
        std::hint::black_box(source);
    }

    #[test]
    #[ignore = "manual deterministic 60MP AI-mask overlap scratch benchmark"]
    fn synthetic_60mp_ai_mask_overlap_scratch_harness() {
        let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(9_504_u32);
        let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(6_336_u32);
        let mode =
            std::env::var("RAW_EDITOR_AI_MASK_BENCH_MODE").unwrap_or_else(|_| "tiled".to_string());
        assert!(matches!(mode.as_str(), "full" | "tiled"));

        let source = GrayImage::from_fn(width, height, |x, y| {
            Luma([x.wrapping_mul(17).wrapping_add(y.wrapping_mul(13)) as u8])
        });
        let mut composition = GrayImage::from_pixel(width, height, Luma([17]));
        let transform = AiMaskTransform::from_options(None, None, None, None);
        let depth_selection = AiDepthSelection::new(18.0, 82.0, 10.0, 10.0, 1.0);
        let grow_feather = GrowFeatherPlan::new(2.0, 1.0, width, height);
        let halo = depth_selection
            .blur_radius()
            .saturating_add(grow_feather.halo());
        let sub_mask = SubMask {
            id: "synthetic-ai-mask".to_string(),
            mask_type: "ai-depth".to_string(),
            visible: true,
            invert: false,
            opacity: 100.0,
            mode: SubMaskMode::Additive,
            parameters: Value::Null,
        };
        let pid = sysinfo::get_current_pid().expect("resolve benchmark process id");
        let mut baseline_system = sysinfo::System::new();
        baseline_system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        let baseline_rss = baseline_system
            .process(pid)
            .expect("read benchmark process after AI source and output allocation")
            .memory();

        let running = Arc::new(AtomicBool::new(true));
        let peak_rss = Arc::new(AtomicU64::new(baseline_rss));
        let sampler_running = Arc::clone(&running);
        let sampler_peak = Arc::clone(&peak_rss);
        let sampler = std::thread::spawn(move || {
            let mut system = sysinfo::System::new();
            while sampler_running.load(Ordering::Relaxed) {
                system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
                if let Some(process) = system.process(pid) {
                    sampler_peak.fetch_max(process.memory(), Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        });

        let started = Instant::now();
        if mode == "full" {
            let mut full_sub_mask =
                transform.rasterize(&source, width, height, 1.0, (0.0, 0.0), (0, 0));
            depth_selection.apply_pointwise(&mut full_sub_mask);
            depth_selection.apply_blur(&mut full_sub_mask);
            grow_feather.apply(&mut full_sub_mask);
            composite_sub_mask_tile(
                &mut composition,
                &full_sub_mask,
                0,
                0,
                SubMaskMode::Additive,
            );
        } else {
            for tile_y in (0..height).step_by(GPU_TILE_SIZE as usize) {
                let tile_height = (height - tile_y).min(GPU_TILE_SIZE);
                for tile_x in (0..width).step_by(GPU_TILE_SIZE as usize) {
                    let tile_width = (width - tile_x).min(GPU_TILE_SIZE);
                    let expanded_x = tile_x.saturating_sub(halo);
                    let expanded_y = tile_y.saturating_sub(halo);
                    let expanded_right = tile_x
                        .saturating_add(tile_width)
                        .saturating_add(halo)
                        .min(width);
                    let expanded_bottom = tile_y
                        .saturating_add(tile_height)
                        .saturating_add(halo)
                        .min(height);
                    let mut tile = transform.rasterize(
                        &source,
                        expanded_right - expanded_x,
                        expanded_bottom - expanded_y,
                        1.0,
                        (0.0, 0.0),
                        (expanded_x, expanded_y),
                    );
                    depth_selection.apply_pointwise(&mut tile);
                    depth_selection.apply_blur(&mut tile);
                    grow_feather.apply(&mut tile);
                    composite_sub_mask_tile_region(
                        &mut composition,
                        &mut tile,
                        tile_x - expanded_x,
                        tile_y - expanded_y,
                        tile_width,
                        tile_height,
                        tile_x,
                        tile_y,
                        &sub_mask,
                    );
                }
            }
        }
        let elapsed = started.elapsed();
        std::thread::sleep(Duration::from_millis(10));
        running.store(false, Ordering::Relaxed);
        sampler.join().expect("join AI-mask RSS sampler");

        let sample_stride = (composition.as_raw().len() / 4_096).max(1);
        let sample_hash = composition
            .as_raw()
            .iter()
            .step_by(sample_stride)
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, sample| {
                (hash ^ u64::from(*sample)).wrapping_mul(0x0000_0100_0000_01b3)
            });
        let pixel_bytes = u64::from(width) * u64::from(height);
        let peak_rss = peak_rss.load(Ordering::Relaxed);
        let expected_scratch_bytes = if mode == "full" {
            pixel_bytes * 3
        } else {
            let expanded_width = width.min(GPU_TILE_SIZE.saturating_add(halo * 2));
            let expanded_height = height.min(GPU_TILE_SIZE.saturating_add(halo * 2));
            u64::from(expanded_width) * u64::from(expanded_height) * 3
        };
        println!(
            "{{\"mode\":\"{}\",\"width\":{},\"height\":{},\"haloPixels\":{},\"elapsedMs\":{},\"baselineRssBytes\":{},\"peakRssBytes\":{},\"peakDeltaBytes\":{},\"sourceAndOutputBytes\":{},\"expectedScratchBytes\":{},\"sampleHash\":\"{:016x}\"}}",
            mode,
            width,
            height,
            halo,
            elapsed.as_millis(),
            baseline_rss,
            peak_rss,
            peak_rss.saturating_sub(baseline_rss),
            pixel_bytes * 2,
            expected_scratch_bytes,
            sample_hash,
        );
        std::hint::black_box(source);
        std::hint::black_box(composition);
    }

    #[test]
    #[ignore = "manual deterministic 60MP mask composition scratch benchmark"]
    fn synthetic_60mp_mask_composition_scratch_harness() {
        let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(9_504_u32);
        let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(6_336_u32);
        let mode = std::env::var("RAW_EDITOR_MASK_COMPOSITION_BENCH_MODE")
            .unwrap_or_else(|_| "tiled".to_string());
        assert!(matches!(mode.as_str(), "full" | "tiled"));

        let mut composition = GrayImage::from_pixel(width, height, Luma([17]));
        let pid = sysinfo::get_current_pid().expect("resolve benchmark process id");
        let mut baseline_system = sysinfo::System::new();
        baseline_system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        let baseline_rss = baseline_system
            .process(pid)
            .expect("read benchmark process after output allocation")
            .memory();

        let running = Arc::new(AtomicBool::new(true));
        let peak_rss = Arc::new(AtomicU64::new(baseline_rss));
        let sampler_running = Arc::clone(&running);
        let sampler_peak = Arc::clone(&peak_rss);
        let sampler = std::thread::spawn(move || {
            let mut system = sysinfo::System::new();
            while sampler_running.load(Ordering::Relaxed) {
                system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
                if let Some(process) = system.process(pid) {
                    sampler_peak.fetch_max(process.memory(), Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        });

        let started = Instant::now();
        if mode == "full" {
            let full_sub_mask = GrayImage::from_fn(width, height, |x, y| {
                Luma([x.wrapping_mul(17).wrapping_add(y * 13) as u8])
            });
            std::thread::sleep(Duration::from_millis(10));
            let mut output = Some(composition);
            composite_sub_mask(&mut output, full_sub_mask, SubMaskMode::Additive);
            composition = output.expect("full-frame composition output");
        } else {
            for tile_y in (0..height).step_by(GPU_TILE_SIZE as usize) {
                let tile_height = (height - tile_y).min(GPU_TILE_SIZE);
                for tile_x in (0..width).step_by(GPU_TILE_SIZE as usize) {
                    let tile_width = (width - tile_x).min(GPU_TILE_SIZE);
                    let tile = GrayImage::from_fn(tile_width, tile_height, |x, y| {
                        Luma([(x + tile_x)
                            .wrapping_mul(17)
                            .wrapping_add((y + tile_y) * 13) as u8])
                    });
                    if tile_x == 0 && tile_y == 0 {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    composite_sub_mask_tile(
                        &mut composition,
                        &tile,
                        tile_x,
                        tile_y,
                        SubMaskMode::Additive,
                    );
                }
            }
        }
        let elapsed = started.elapsed();
        std::thread::sleep(Duration::from_millis(10));
        running.store(false, Ordering::Relaxed);
        sampler.join().expect("join mask composition RSS sampler");

        let sample_stride = (composition.as_raw().len() / 4_096).max(1);
        let sample_hash = composition
            .as_raw()
            .iter()
            .step_by(sample_stride)
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, sample| {
                (hash ^ u64::from(*sample)).wrapping_mul(0x0000_0100_0000_01b3)
            });
        let peak_rss = peak_rss.load(Ordering::Relaxed);
        let expected_scratch_bytes = if mode == "full" {
            u64::from(width) * u64::from(height)
        } else {
            u64::from(width.min(GPU_TILE_SIZE)) * u64::from(height.min(GPU_TILE_SIZE))
        };
        println!(
            "{{\"mode\":\"{}\",\"width\":{},\"height\":{},\"elapsedMs\":{},\"baselineRssBytes\":{},\"peakRssBytes\":{},\"peakDeltaBytes\":{},\"expectedScratchBytes\":{},\"sampleHash\":\"{:016x}\"}}",
            mode,
            width,
            height,
            elapsed.as_millis(),
            baseline_rss,
            peak_rss,
            peak_rss.saturating_sub(baseline_rss),
            expected_scratch_bytes,
            sample_hash,
        );
        std::hint::black_box(composition);
    }

    #[test]
    #[ignore = "manual deterministic 60MP mask cache ownership benchmark"]
    fn synthetic_60mp_mask_cache_ownership_harness() {
        enum Fixture {
            Cloned {
                cache: GrayImage,
                caller: GrayImage,
            },
            Shared {
                cache: Arc<GrayImage>,
                caller: Arc<GrayImage>,
            },
        }

        let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(9_504_u32);
        let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(6_336_u32);
        let mode =
            std::env::var("RAW_EDITOR_MASK_BENCH_MODE").unwrap_or_else(|_| "shared".to_string());
        assert!(matches!(mode.as_str(), "shared" | "cloned"));

        let pid = sysinfo::get_current_pid().expect("resolve benchmark process id");
        let mut baseline_system = sysinfo::System::new();
        baseline_system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        let baseline_rss = baseline_system
            .process(pid)
            .expect("read benchmark process before mask allocation")
            .memory();

        let running = Arc::new(AtomicBool::new(true));
        let peak_rss = Arc::new(AtomicU64::new(baseline_rss));
        let sampler_running = Arc::clone(&running);
        let sampler_peak = Arc::clone(&peak_rss);
        let sampler = std::thread::spawn(move || {
            let mut system = sysinfo::System::new();
            while sampler_running.load(Ordering::Relaxed) {
                system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
                if let Some(process) = system.process(pid) {
                    sampler_peak.fetch_max(process.memory(), Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        });

        let started = Instant::now();
        let source = GrayImage::from_fn(width, height, |x, y| {
            Luma([x.wrapping_mul(17).wrapping_add(y * 13) as u8])
        });
        let fixture = if mode == "cloned" {
            Fixture::Cloned {
                caller: source.clone(),
                cache: source,
            }
        } else {
            let cache = Arc::new(source);
            Fixture::Shared {
                caller: Arc::clone(&cache),
                cache,
            }
        };
        let elapsed = started.elapsed();
        std::thread::sleep(Duration::from_millis(25));
        running.store(false, Ordering::Relaxed);
        sampler.join().expect("join mask RSS sampler");

        let (cache_raw, caller_raw, shared_allocation) = match &fixture {
            Fixture::Cloned { cache, caller } => (cache.as_raw(), caller.as_raw(), false),
            Fixture::Shared { cache, caller } => {
                assert!(Arc::ptr_eq(cache, caller));
                (cache.as_raw(), caller.as_raw(), true)
            }
        };
        assert_eq!(cache_raw, caller_raw);
        let sample_stride = (cache_raw.len() / 4_096).max(1);
        let sample_hash = cache_raw
            .iter()
            .step_by(sample_stride)
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, sample| {
                (hash ^ u64::from(*sample)).wrapping_mul(0x0000_0100_0000_01b3)
            });
        let mask_bytes = u64::from(width) * u64::from(height);
        let expected_live_pixel_bytes = mask_bytes * if shared_allocation { 1 } else { 2 };
        println!(
            "{{\"mode\":\"{}\",\"width\":{},\"height\":{},\"elapsedMs\":{},\"baselineRssBytes\":{},\"peakRssBytes\":{},\"peakDeltaBytes\":{},\"maskBytes\":{},\"expectedLivePixelBytes\":{},\"sampleHash\":\"{:016x}\"}}",
            mode,
            width,
            height,
            elapsed.as_millis(),
            baseline_rss,
            peak_rss.load(Ordering::Relaxed),
            peak_rss
                .load(Ordering::Relaxed)
                .saturating_sub(baseline_rss),
            mask_bytes,
            expected_live_pixel_bytes,
            sample_hash,
        );
        std::hint::black_box(fixture);
    }
}
