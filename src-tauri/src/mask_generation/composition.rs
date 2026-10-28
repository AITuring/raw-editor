//! Sub-mask generation and mask composition.
//!
//! Generates each sub-mask, applies its invert and opacity and combines it
//! additively, subtractively or by intersection, in bounded tiles when a
//! full-frame temporary is not needed.

use crate::render_strategy::GPU_TILE_SIZE;
use image::GrayImage;
use image::Luma;
use serde_json::Value;

use super::{
    BrushMaskParameters, FlowMaskParameters, GrowFeatherPlan, LinearMaskParameters, MaskDefinition,
    ParametricMaskParameters, RadialMaskParameters, RangeMaskSource, SubMask, SubMaskMode,
    TiledAiMaskRasterizer, generate_ai_depth_bitmap, generate_ai_foreground_bitmap,
    generate_ai_sky_bitmap, generate_ai_subject_bitmap, generate_brush_bitmap,
    generate_color_bitmap, generate_color_bitmap_from_parameters, generate_flow_bitmap,
    generate_linear_bitmap, generate_luminance_bitmap, generate_luminance_bitmap_from_parameters,
    generate_radial_bitmap,
};

pub(super) fn generate_all_bitmap(width: u32, height: u32) -> GrayImage {
    GrayImage::from_pixel(width, height, Luma([255]))
}

pub(super) fn generate_sub_mask_bitmap(
    sub_mask: &SubMask,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    range_source: Option<&dyn RangeMaskSource>,
) -> Option<GrayImage> {
    if !sub_mask.visible {
        return None;
    }

    match sub_mask.mask_type.as_str() {
        "radial" => {
            let parameters =
                serde_json::from_value(sub_mask.parameters.clone()).unwrap_or_default();
            Some(generate_radial_bitmap(
                &parameters,
                width,
                height,
                scale,
                crop_offset,
                (0, 0),
            ))
        }
        "linear" => {
            let parameters =
                serde_json::from_value(sub_mask.parameters.clone()).unwrap_or_default();
            Some(generate_linear_bitmap(
                &parameters,
                width,
                height,
                scale,
                crop_offset,
                (0, 0),
            ))
        }
        "brush" | "clone" | "heal" => {
            let parameters =
                serde_json::from_value(sub_mask.parameters.clone()).unwrap_or_default();
            Some(generate_brush_bitmap(
                &parameters,
                width,
                height,
                scale,
                crop_offset,
                (0, 0),
            ))
        }
        "flow" => {
            let parameters =
                serde_json::from_value(sub_mask.parameters.clone()).unwrap_or_default();
            Some(generate_flow_bitmap(
                &parameters,
                width,
                height,
                scale,
                crop_offset,
                (0, 0),
            ))
        }
        "color" => generate_color_bitmap(
            &sub_mask.parameters,
            width,
            height,
            scale,
            crop_offset,
            range_source,
        ),
        "luminance" => generate_luminance_bitmap(
            &sub_mask.parameters,
            width,
            height,
            scale,
            crop_offset,
            range_source,
        ),
        "ai-subject" => {
            generate_ai_subject_bitmap(&sub_mask.parameters, width, height, scale, crop_offset)
        }
        "ai-foreground" => {
            generate_ai_foreground_bitmap(&sub_mask.parameters, width, height, scale, crop_offset)
        }
        "ai-sky" => generate_ai_sky_bitmap(&sub_mask.parameters, width, height, scale, crop_offset),
        "ai-depth" => {
            generate_ai_depth_bitmap(&sub_mask.parameters, width, height, scale, crop_offset)
        }
        "quick-eraser" => {
            generate_ai_subject_bitmap(&sub_mask.parameters, width, height, scale, crop_offset)
        }
        "all" => Some(generate_all_bitmap(width, height)),
        _ => None,
    }
}

pub(super) enum TiledSubMaskRasterizer {
    Radial(RadialMaskParameters),
    Linear(LinearMaskParameters),
    Brush(BrushMaskParameters),
    Flow(FlowMaskParameters),
    Color(ParametricMaskParameters),
    Luminance(ParametricMaskParameters),
    Ai(TiledAiMaskRasterizer),
    All,
}

impl TiledSubMaskRasterizer {
    pub(super) fn new(sub_mask: &SubMask) -> Option<Self> {
        match sub_mask.mask_type.as_str() {
            "radial" => Some(Self::Radial(
                serde_json::from_value(sub_mask.parameters.clone()).unwrap_or_default(),
            )),
            "linear" => Some(Self::Linear(
                serde_json::from_value(sub_mask.parameters.clone()).unwrap_or_default(),
            )),
            "brush" | "clone" | "heal" => Some(Self::Brush(
                serde_json::from_value(sub_mask.parameters.clone()).unwrap_or_default(),
            )),
            "flow" => Some(Self::Flow(
                serde_json::from_value(sub_mask.parameters.clone()).unwrap_or_default(),
            )),
            "color" => Some(Self::Color(
                serde_json::from_value(sub_mask.parameters.clone()).ok()?,
            )),
            "luminance" => Some(Self::Luminance(
                serde_json::from_value(sub_mask.parameters.clone()).ok()?,
            )),
            "ai-subject" | "ai-foreground" | "ai-sky" | "ai-depth" | "quick-eraser" => {
                Some(Self::Ai(TiledAiMaskRasterizer::new(sub_mask)?))
            }
            "all" => Some(Self::All),
            _ => None,
        }
    }

    pub(super) fn rasterize(
        &self,
        width: u32,
        height: u32,
        scale: f32,
        crop_offset: (f32, f32),
        output_origin: (u32, u32),
        range_source: Option<&dyn RangeMaskSource>,
    ) -> Option<GrayImage> {
        match self {
            Self::Radial(parameters) => Some(generate_radial_bitmap(
                parameters,
                width,
                height,
                scale,
                crop_offset,
                output_origin,
            )),
            Self::Linear(parameters) => Some(generate_linear_bitmap(
                parameters,
                width,
                height,
                scale,
                crop_offset,
                output_origin,
            )),
            Self::Brush(parameters) => Some(generate_brush_bitmap(
                parameters,
                width,
                height,
                scale,
                crop_offset,
                output_origin,
            )),
            Self::Flow(parameters) => Some(generate_flow_bitmap(
                parameters,
                width,
                height,
                scale,
                crop_offset,
                output_origin,
            )),
            Self::Color(parameters) => generate_color_bitmap_from_parameters(
                parameters,
                width,
                height,
                scale,
                crop_offset,
                output_origin,
                range_source?,
            ),
            Self::Luminance(parameters) => generate_luminance_bitmap_from_parameters(
                parameters,
                width,
                height,
                scale,
                crop_offset,
                output_origin,
                range_source?,
            ),
            Self::Ai(rasterizer) => {
                Some(rasterizer.rasterize(width, height, scale, crop_offset, output_origin))
            }
            Self::All => Some(generate_all_bitmap(width, height)),
        }
    }

    pub(super) fn halo(&self, width: u32, height: u32) -> u32 {
        match self {
            Self::Color(parameters) | Self::Luminance(parameters) => {
                GrowFeatherPlan::new(parameters.grow, parameters.feather, width, height).halo()
            }
            Self::Ai(rasterizer) => rasterizer.halo(width, height),
            _ => 0,
        }
    }

    pub(super) fn apply_cross_pixel_filters(&self, mask: &mut GrayImage, width: u32, height: u32) {
        match self {
            Self::Color(parameters) | Self::Luminance(parameters) => {
                GrowFeatherPlan::new(parameters.grow, parameters.feather, width, height)
                    .apply(mask);
            }
            Self::Ai(rasterizer) => {
                rasterizer.apply_cross_pixel_filters(mask, width, height);
            }
            _ => {}
        }
    }

    pub(super) fn is_ready(&self, range_source: Option<&dyn RangeMaskSource>) -> bool {
        match self {
            Self::Color(parameters) | Self::Luminance(parameters) => {
                range_source.is_some_and(|source| {
                    let (width, height) = source.source_dimensions();
                    let target_x = parameters.target_x.round() as i32;
                    let target_y = parameters.target_y.round() as i32;
                    target_x >= 0
                        && target_y >= 0
                        && target_x < width as i32
                        && target_y < height as i32
                })
            }
            Self::Ai(_) => true,
            _ => true,
        }
    }
}

pub fn generate_mask_bitmap(
    mask_def: &MaskDefinition,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    range_source: Option<&dyn RangeMaskSource>,
) -> Option<GrayImage> {
    if !mask_def.visible || mask_def.sub_masks.is_empty() {
        return None;
    }

    let mut final_mask: Option<GrayImage> = None;

    for sub_mask in &mask_def.sub_masks {
        // The historical composition starts from black. Subtractive and intersect masks cannot
        // change that state, so defer the first allocation/generation until an additive mask can
        // become the output buffer itself.
        if final_mask.is_none() && sub_mask.mode != SubMaskMode::Additive {
            continue;
        }
        if should_generate_sub_mask_in_tiles(sub_mask, final_mask.is_some(), width, height)
            && composite_sub_mask_tiled(
                &mut final_mask,
                sub_mask,
                width,
                height,
                scale,
                crop_offset,
                range_source,
            )
        {
            continue;
        }
        if let Some(mut sub_bitmap) =
            generate_sub_mask_bitmap(sub_mask, width, height, scale, crop_offset, range_source)
        {
            apply_sub_mask_modifiers(&mut sub_bitmap, sub_mask);
            composite_sub_mask(&mut final_mask, sub_bitmap, sub_mask.mode);
        }
    }

    let mut final_mask = final_mask.unwrap_or_else(|| GrayImage::new(width, height));

    if mask_def.invert {
        for pixel in final_mask.pixels_mut() {
            pixel[0] = 255 - pixel[0];
        }
    }

    let opacity_multiplier = (mask_def.opacity / 100.0).clamp(0.0, 1.0);
    if opacity_multiplier < 1.0 {
        for pixel in final_mask.pixels_mut() {
            pixel[0] = (pixel[0] as f32 * opacity_multiplier) as u8;
        }
    }

    Some(final_mask)
}

pub(super) fn should_generate_sub_mask_in_tiles(
    sub_mask: &SubMask,
    has_composition: bool,
    width: u32,
    height: u32,
) -> bool {
    if !sub_mask.visible {
        return false;
    }
    match sub_mask.mask_type.as_str() {
        // Brush and flow generators otherwise hold their full-frame result plus a stroke layer.
        "brush" | "clone" | "heal" => {
            serde_json::from_value::<BrushMaskParameters>(sub_mask.parameters.clone())
                .is_ok_and(|parameters| parameters.lines.iter().any(|line| !line.points.is_empty()))
        }
        "flow" => serde_json::from_value::<FlowMaskParameters>(sub_mask.parameters.clone())
            .is_ok_and(|parameters| parameters.lines.iter().any(|line| !line.points.is_empty())),
        // The first pointwise additive mask can become the output allocation directly. Later
        // masks use bounded tiles so composition never needs a second full-frame bitmap.
        "radial" | "linear" | "all" => has_composition,
        // A first unfiltered range mask can likewise become the output directly. Cross-pixel
        // grow/feather and every later range mask use an exact finite halo around each tile.
        "color" | "luminance" => {
            serde_json::from_value::<ParametricMaskParameters>(sub_mask.parameters.clone())
                .is_ok_and(|parameters| {
                    has_composition
                        || GrowFeatherPlan::new(parameters.grow, parameters.feather, width, height)
                            .has_effect()
                })
        }
        // AI masks keep their decoded source bitmap, but bounded output/filter tiles avoid a
        // transformed full-frame temporary whenever composition or a cross-pixel filter needs it.
        "ai-subject" | "ai-foreground" | "ai-sky" | "quick-eraser" | "ai-depth" => {
            let grow = sub_mask
                .parameters
                .get("grow")
                .and_then(Value::as_f64)
                .unwrap_or_default() as f32;
            let feather = sub_mask
                .parameters
                .get("feather")
                .and_then(Value::as_f64)
                .unwrap_or_default() as f32;
            let grow_feather_effect =
                GrowFeatherPlan::new(grow, feather, width, height).has_effect();
            let depth_blur_effect = sub_mask.mask_type == "ai-depth" && feather > 0.0;
            has_composition || grow_feather_effect || depth_blur_effect
        }
        _ => false,
    }
}

pub(super) fn apply_sub_mask_modifiers(bitmap: &mut GrayImage, sub_mask: &SubMask) {
    apply_sub_mask_modifiers_to_pixels(bitmap.as_mut(), sub_mask);
}

pub(super) fn apply_sub_mask_modifiers_to_pixels(pixels: &mut [u8], sub_mask: &SubMask) {
    if sub_mask.invert {
        for pixel in pixels.iter_mut() {
            *pixel = 255 - *pixel;
        }
    }

    let opacity_multiplier = (sub_mask.opacity / 100.0).clamp(0.0, 1.0);
    if opacity_multiplier < 1.0 {
        for pixel in pixels {
            *pixel = (*pixel as f32 * opacity_multiplier) as u8;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn composite_sub_mask_tiled(
    final_mask: &mut Option<GrayImage>,
    sub_mask: &SubMask,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    range_source: Option<&dyn RangeMaskSource>,
) -> bool {
    composite_sub_mask_tiled_with_edge(
        final_mask,
        sub_mask,
        width,
        height,
        scale,
        crop_offset,
        range_source,
        GPU_TILE_SIZE,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn composite_sub_mask_tiled_with_edge(
    final_mask: &mut Option<GrayImage>,
    sub_mask: &SubMask,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
    range_source: Option<&dyn RangeMaskSource>,
    tile_edge: u32,
) -> bool {
    debug_assert!(tile_edge > 0);
    debug_assert!(final_mask.is_some() || sub_mask.mode == SubMaskMode::Additive);
    let Some(rasterizer) = TiledSubMaskRasterizer::new(sub_mask) else {
        return false;
    };
    if !rasterizer.is_ready(range_source) {
        return false;
    }
    let halo = rasterizer.halo(width, height);
    let output = final_mask.get_or_insert_with(|| GrayImage::new(width, height));

    for tile_y in (0..height).step_by(tile_edge as usize) {
        let tile_height = (height - tile_y).min(tile_edge);
        for tile_x in (0..width).step_by(tile_edge as usize) {
            let tile_width = (width - tile_x).min(tile_edge);
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
            let expanded_width = expanded_right - expanded_x;
            let expanded_height = expanded_bottom - expanded_y;
            let mut tile = rasterizer
                .rasterize(
                    expanded_width,
                    expanded_height,
                    scale,
                    crop_offset,
                    (expanded_x, expanded_y),
                    range_source,
                )
                .expect("validated tiled mask rasterizer");
            rasterizer.apply_cross_pixel_filters(&mut tile, width, height);
            composite_sub_mask_tile_region(
                output,
                &mut tile,
                tile_x - expanded_x,
                tile_y - expanded_y,
                tile_width,
                tile_height,
                tile_x,
                tile_y,
                sub_mask,
            );
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
pub(super) fn composite_sub_mask_tile_region(
    final_mask: &mut GrayImage,
    tile: &mut GrayImage,
    source_x: u32,
    source_y: u32,
    width: u32,
    height: u32,
    tile_x: u32,
    tile_y: u32,
    sub_mask: &SubMask,
) {
    let (output_width, output_height) = final_mask.dimensions();
    let (source_width, source_height) = tile.dimensions();
    debug_assert!(tile_x.saturating_add(width) <= output_width);
    debug_assert!(tile_y.saturating_add(height) <= output_height);
    debug_assert!(source_x.saturating_add(width) <= source_width);
    debug_assert!(source_y.saturating_add(height) <= source_height);

    let output_width = output_width as usize;
    let source_width = source_width as usize;
    let width = width as usize;
    for row in 0..height as usize {
        let output_start = (tile_y as usize + row) * output_width + tile_x as usize;
        let source_start = (source_y as usize + row) * source_width + source_x as usize;
        let source_pixels = &mut tile.as_mut()[source_start..source_start + width];
        apply_sub_mask_modifiers_to_pixels(source_pixels, sub_mask);
        composite_mask_pixels(
            &mut final_mask.as_mut()[output_start..output_start + width],
            source_pixels,
            sub_mask.mode,
        );
    }
}

#[cfg(test)]
pub(super) fn composite_sub_mask_tile(
    final_mask: &mut GrayImage,
    tile: &GrayImage,
    tile_x: u32,
    tile_y: u32,
    mode: SubMaskMode,
) {
    let (output_width, output_height) = final_mask.dimensions();
    let (tile_width, tile_height) = tile.dimensions();
    debug_assert!(tile_x.saturating_add(tile_width) <= output_width);
    debug_assert!(tile_y.saturating_add(tile_height) <= output_height);

    let output_width = output_width as usize;
    let tile_width = tile_width as usize;
    for row in 0..tile_height as usize {
        let output_start = (tile_y as usize + row) * output_width + tile_x as usize;
        let tile_start = row * tile_width;
        composite_mask_pixels(
            &mut final_mask.as_mut()[output_start..output_start + tile_width],
            &tile.as_raw()[tile_start..tile_start + tile_width],
            mode,
        );
    }
}

pub(super) fn composite_sub_mask(
    final_mask: &mut Option<GrayImage>,
    sub_bitmap: GrayImage,
    mode: SubMaskMode,
) {
    if final_mask.is_none() {
        if mode == SubMaskMode::Additive {
            *final_mask = Some(sub_bitmap);
        }
        return;
    }

    debug_assert_eq!(
        final_mask
            .as_ref()
            .expect("mask allocation checked above")
            .dimensions(),
        sub_bitmap.dimensions()
    );
    let final_raw = final_mask
        .as_mut()
        .expect("mask allocation checked above")
        .as_mut();
    composite_mask_pixels(final_raw, sub_bitmap.as_raw(), mode);
}

pub(super) fn composite_mask_pixels(output: &mut [u8], input: &[u8], mode: SubMaskMode) {
    debug_assert_eq!(output.len(), input.len());
    match mode {
        SubMaskMode::Additive => output
            .iter_mut()
            .zip(input)
            .for_each(|(output, input)| *output = (*output).max(*input)),
        SubMaskMode::Subtractive => output
            .iter_mut()
            .zip(input)
            .for_each(|(output, input)| *output = output.saturating_sub(*input)),
        SubMaskMode::Intersect => output
            .iter_mut()
            .zip(input)
            .for_each(|(output, input)| *output = (*output).min(*input)),
    }
}
