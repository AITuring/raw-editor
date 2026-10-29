//! AI patch compositing on the base image.
//!
//! Visible AI patches are blended into the decoded image through their sub-mask
//! bitmaps; load_and_composite decodes and composites in one call.

use crate::app_settings::AppSettings;
use crate::color_management::srgb_to_linear_channel;
use crate::mask_generation::MaskDefinition;
use crate::mask_generation::SubMask;
use crate::mask_generation::generate_mask_bitmap;
use anyhow::Context;
use anyhow::Result;
use base64::Engine as _;
use base64::engine::general_purpose;
use image::DynamicImage;
use image::GenericImageView;
use image::imageops;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::AtomicUsize;

use super::load_base_image_from_bytes_with_exif_persistence;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct PatchMaskInfo {
    pub(super) id: String,
    pub(super) name: String,
    #[serde(default)]
    pub(super) invert: bool,
    #[serde(default)]
    pub(super) sub_masks: Vec<SubMask>,
}

pub(super) fn srgb_to_linear_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = [0.0f32; 256];
        for (i, v) in lut.iter_mut().enumerate() {
            *v = srgb_to_linear_channel(i as f32 / 255.0);
        }
        lut
    })
}

pub fn load_and_composite(
    base_image: &[u8],
    path: &str,
    adjustments: &Value,
    use_fast_raw_dev: bool,
    settings: &AppSettings,
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
) -> Result<DynamicImage> {
    load_and_composite_with_exif_persistence(
        base_image,
        path,
        adjustments,
        use_fast_raw_dev,
        settings,
        cancel_token,
        true,
    )
}

pub(super) fn load_and_composite_with_exif_persistence(
    base_image: &[u8],
    path: &str,
    adjustments: &Value,
    use_fast_raw_dev: bool,
    settings: &AppSettings,
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
    persist_exif: bool,
) -> Result<DynamicImage> {
    let base_image = load_base_image_from_bytes_with_exif_persistence(
        base_image,
        path,
        use_fast_raw_dev,
        settings,
        cancel_token,
        persist_exif,
    )?;
    composite_patches_on_image(&base_image, adjustments)
}

pub fn composite_patches_on_image(
    base_image: &DynamicImage,
    current_adjustments: &Value,
) -> Result<DynamicImage> {
    let patches_val = match current_adjustments.get("aiPatches") {
        Some(val) => val,
        None => return Ok(base_image.clone()),
    };

    let patches_arr = match patches_val.as_array() {
        Some(arr) if !arr.is_empty() => arr,
        _ => return Ok(base_image.clone()),
    };

    let visible_patches: Vec<&Value> = patches_arr
        .par_iter()
        .filter(|patch_obj| {
            let is_visible = patch_obj
                .get("visible")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            if !is_visible {
                return false;
            }
            patch_obj
                .get("patchData")
                .and_then(|data| data.get("color"))
                .and_then(|color| color.as_str())
                .is_some_and(|s| !s.is_empty())
        })
        .collect();

    if visible_patches.is_empty() {
        return Ok(base_image.clone());
    }

    let (base_w, base_h) = base_image.dimensions();

    struct DecodedPatch {
        offset_x: Option<u32>,
        offset_y: Option<u32>,
        mask: image::GrayImage,
        color: image::RgbImage,
        is_srgb_encoded: bool,
    }

    let decoded_patches: Result<Vec<DecodedPatch>> = visible_patches
        .par_iter()
        .map(|patch_obj| {
            let patch_data = patch_obj.get("patchData").context("Missing patchData")?;
            let offset_x = patch_data
                .get("offsetX")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            let offset_y = patch_data
                .get("offsetY")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            let is_cropped = offset_x.is_some() && offset_y.is_some();

            let is_srgb_encoded = patch_data
                .get("isSrgbEncoded")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let mask_bitmap = if let Some(mask_b64) = patch_data
                .get("mask")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
            {
                let mask_bytes = general_purpose::STANDARD.decode(mask_b64)?;
                let mask_img = image::load_from_memory(&mask_bytes)?.to_luma8();
                if !is_cropped && (mask_img.width() != base_w || mask_img.height() != base_h) {
                    imageops::resize(&mask_img, base_w, base_h, imageops::FilterType::Lanczos3)
                } else {
                    mask_img
                }
            } else {
                let patch_info: PatchMaskInfo = serde_json::from_value((*patch_obj).clone())
                    .context("Failed to deserialize patch info for mask generation")?;

                let mask_def = MaskDefinition {
                    id: patch_info.id,
                    name: patch_info.name,
                    visible: true,
                    invert: patch_info.invert,
                    opacity: 100.0,
                    adjustments: Value::Null,
                    sub_masks: patch_info.sub_masks,
                };

                let orientation_steps = current_adjustments
                    .get("orientationSteps")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u8;
                let (trans_w, trans_h) = if orientation_steps % 2 == 1 {
                    (base_h, base_w)
                } else {
                    (base_w, base_h)
                };

                let mut gen_mask =
                    generate_mask_bitmap(&mask_def, trans_w, trans_h, 1.0, (0.0, 0.0), None)
                        .context("Failed to generate mask from sub_masks for compositing")?;

                gen_mask =
                    crate::image_processing::inverse_transform_mask(gen_mask, current_adjustments);

                if let (Some(ox), Some(oy)) = (offset_x, offset_y) {
                    let w = patch_data
                        .get("width")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as u32)
                        .unwrap_or(base_w);
                    let h = patch_data
                        .get("height")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as u32)
                        .unwrap_or(base_h);
                    let crop_w = w.min(base_w.saturating_sub(ox));
                    let crop_h = h.min(base_h.saturating_sub(oy));
                    gen_mask = imageops::crop_imm(&gen_mask, ox, oy, crop_w, crop_h).to_image();
                }
                gen_mask
            };

            let color_b64 = patch_data
                .get("color")
                .and_then(|v| v.as_str())
                .context("Missing color data")?;
            let color_bytes = general_purpose::STANDARD.decode(color_b64)?;
            let color_image_u8 = image::load_from_memory(&color_bytes)?.to_rgb8();

            let (patch_w, patch_h) = color_image_u8.dimensions();
            let final_color = if !is_cropped && (base_w != patch_w || base_h != patch_h) {
                imageops::resize(
                    &color_image_u8,
                    base_w,
                    base_h,
                    imageops::FilterType::Lanczos3,
                )
            } else {
                color_image_u8
            };

            Ok(DecodedPatch {
                offset_x,
                offset_y,
                mask: mask_bitmap,
                color: final_color,
                is_srgb_encoded,
            })
        })
        .collect();

    let decoded_patches = decoded_patches?;

    let mut composited_image = base_image.clone();
    let lut = srgb_to_linear_lut();

    let get_color = |patch: &DecodedPatch, r: u8, g: u8, b: u8| -> (f32, f32, f32) {
        if patch.is_srgb_encoded {
            (lut[r as usize], lut[g as usize], lut[b as usize])
        } else {
            (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
        }
    };

    match &mut composited_image {
        DynamicImage::ImageRgb32F(img_buf) => {
            for patch in decoded_patches {
                if let (Some(ox), Some(oy)) = (patch.offset_x, patch.offset_y) {
                    let max_x = (ox + patch.mask.width()).min(base_w);
                    let max_y = (oy + patch.mask.height()).min(base_h);

                    for y in oy..max_y {
                        let py = y - oy;
                        for x in ox..max_x {
                            let px = x - ox;
                            let mask_value = patch.mask.get_pixel(px, py)[0];
                            if mask_value > 0 {
                                let patch_pixel = patch.color.get_pixel(px, py);
                                let (pr, pg, pb) = get_color(
                                    &patch,
                                    patch_pixel[0],
                                    patch_pixel[1],
                                    patch_pixel[2],
                                );

                                let alpha = mask_value as f32 / 255.0;
                                let one_minus_alpha = 1.0 - alpha;

                                let base_px = img_buf.get_pixel_mut(x, y);
                                base_px[0] = pr * alpha + base_px[0] * one_minus_alpha;
                                base_px[1] = pg * alpha + base_px[1] * one_minus_alpha;
                                base_px[2] = pb * alpha + base_px[2] * one_minus_alpha;
                            }
                        }
                    }
                } else {
                    img_buf
                        .par_chunks_mut((base_w * 3) as usize)
                        .enumerate()
                        .for_each(|(y, row)| {
                            for x in 0..base_w as usize {
                                let mask_value = patch.mask.get_pixel(x as u32, y as u32)[0];
                                if mask_value > 0 {
                                    let patch_pixel = patch.color.get_pixel(x as u32, y as u32);
                                    let (pr, pg, pb) = get_color(
                                        &patch,
                                        patch_pixel[0],
                                        patch_pixel[1],
                                        patch_pixel[2],
                                    );

                                    let alpha = mask_value as f32 / 255.0;
                                    let one_minus_alpha = 1.0 - alpha;

                                    row[x * 3] = pr * alpha + row[x * 3] * one_minus_alpha;
                                    row[x * 3 + 1] = pg * alpha + row[x * 3 + 1] * one_minus_alpha;
                                    row[x * 3 + 2] = pb * alpha + row[x * 3 + 2] * one_minus_alpha;
                                }
                            }
                        });
                }
            }
        }
        DynamicImage::ImageRgba32F(img_buf) => {
            for patch in decoded_patches {
                if let (Some(ox), Some(oy)) = (patch.offset_x, patch.offset_y) {
                    let max_x = (ox + patch.mask.width()).min(base_w);
                    let max_y = (oy + patch.mask.height()).min(base_h);

                    for y in oy..max_y {
                        let py = y - oy;
                        for x in ox..max_x {
                            let px = x - ox;
                            let mask_value = patch.mask.get_pixel(px, py)[0];
                            if mask_value > 0 {
                                let patch_pixel = patch.color.get_pixel(px, py);
                                let (pr, pg, pb) = get_color(
                                    &patch,
                                    patch_pixel[0],
                                    patch_pixel[1],
                                    patch_pixel[2],
                                );

                                let alpha = mask_value as f32 / 255.0;
                                let one_minus_alpha = 1.0 - alpha;

                                let base_px = img_buf.get_pixel_mut(x, y);
                                base_px[0] = pr * alpha + base_px[0] * one_minus_alpha;
                                base_px[1] = pg * alpha + base_px[1] * one_minus_alpha;
                                base_px[2] = pb * alpha + base_px[2] * one_minus_alpha;
                            }
                        }
                    }
                } else {
                    img_buf
                        .par_chunks_mut((base_w * 4) as usize)
                        .enumerate()
                        .for_each(|(y, row)| {
                            for x in 0..base_w as usize {
                                let mask_value = patch.mask.get_pixel(x as u32, y as u32)[0];
                                if mask_value > 0 {
                                    let patch_pixel = patch.color.get_pixel(x as u32, y as u32);
                                    let (pr, pg, pb) = get_color(
                                        &patch,
                                        patch_pixel[0],
                                        patch_pixel[1],
                                        patch_pixel[2],
                                    );

                                    let alpha = mask_value as f32 / 255.0;
                                    let one_minus_alpha = 1.0 - alpha;

                                    row[x * 4] = pr * alpha + row[x * 4] * one_minus_alpha;
                                    row[x * 4 + 1] = pg * alpha + row[x * 4 + 1] * one_minus_alpha;
                                    row[x * 4 + 2] = pb * alpha + row[x * 4 + 2] * one_minus_alpha;
                                }
                            }
                        });
                }
            }
        }
        _ => {
            let mut rgba32_img = composited_image.to_rgba32f();
            for patch in decoded_patches {
                if let (Some(ox), Some(oy)) = (patch.offset_x, patch.offset_y) {
                    let max_x = (ox + patch.mask.width()).min(base_w);
                    let max_y = (oy + patch.mask.height()).min(base_h);
                    for y in oy..max_y {
                        let py = y - oy;
                        for x in ox..max_x {
                            let px = x - ox;
                            let mask_val = patch.mask.get_pixel(px, py)[0];
                            if mask_val > 0 {
                                let patch_px = patch.color.get_pixel(px, py);
                                let (pr, pg, pb) =
                                    get_color(&patch, patch_px[0], patch_px[1], patch_px[2]);

                                let alpha = mask_val as f32 / 255.0;
                                let one_minus_alpha = 1.0 - alpha;
                                let base_px = rgba32_img.get_pixel_mut(x, y);
                                base_px[0] = pr * alpha + base_px[0] * one_minus_alpha;
                                base_px[1] = pg * alpha + base_px[1] * one_minus_alpha;
                                base_px[2] = pb * alpha + base_px[2] * one_minus_alpha;
                            }
                        }
                    }
                } else {
                    for y in 0..base_h {
                        for x in 0..base_w {
                            let mask_val = patch.mask.get_pixel(x, y)[0];
                            if mask_val > 0 {
                                let patch_px = patch.color.get_pixel(x, y);
                                let (pr, pg, pb) =
                                    get_color(&patch, patch_px[0], patch_px[1], patch_px[2]);

                                let alpha = mask_val as f32 / 255.0;
                                let one_minus_alpha = 1.0 - alpha;
                                let base_px = rgba32_img.get_pixel_mut(x, y);
                                base_px[0] = pr * alpha + base_px[0] * one_minus_alpha;
                                base_px[1] = pg * alpha + base_px[1] * one_minus_alpha;
                                base_px[2] = pb * alpha + base_px[2] * one_minus_alpha;
                            }
                        }
                    }
                }
            }
            composited_image = DynamicImage::ImageRgba32F(rgba32_img);
        }
    }

    Ok(composited_image)
}
