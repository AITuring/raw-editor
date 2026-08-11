use image::DynamicImage;
use std::borrow::Cow;
use std::collections::HashMap;

use crate::app_state::AppState;
use crate::image_processing::{
    Crop, IntoCowImage, apply_coarse_rotation, apply_crop, apply_flip, apply_geometry_warp,
    apply_rotation,
};

pub fn hydrate_sub_masks(
    sub_masks: &mut Vec<serde_json::Value>,
    cache: &mut HashMap<String, serde_json::Value>,
) {
    for sub_mask in sub_masks {
        let id = sub_mask
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        if id.is_empty() {
            continue;
        }

        if let Some(params) = sub_mask
            .get_mut("parameters")
            .and_then(|p| p.as_object_mut())
        {
            let keys_to_check = ["mask_data_base64", "maskDataBase64"];
            for key in keys_to_check {
                if params.contains_key(key) {
                    let val = params.get(key).unwrap();
                    if !val.is_null() {
                        cache.insert(id.clone(), val.clone());
                    } else {
                        if let Some(cached_data) = cache.get(&id) {
                            params.insert(key.to_string(), cached_data.clone());
                        }
                    }
                }
            }
        }
    }
}

pub fn hydrate_adjustments(state: &tauri::State<AppState>, adjustments: &mut serde_json::Value) {
    let mut cache = state.patch_cache.lock().unwrap();

    if let Some(patches) = adjustments
        .get_mut("aiPatches")
        .and_then(|v| v.as_array_mut())
    {
        for patch in patches {
            let id = patch
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();

            if !id.is_empty() {
                let has_data = patch.get("patchData").is_some_and(|v| !v.is_null());

                if has_data {
                    if let Some(data) = patch.get("patchData") {
                        cache.insert(id.clone(), data.clone());
                    }
                } else {
                    if let Some(cached_data) = cache.get(&id) {
                        patch["patchData"] = cached_data.clone();
                    }
                }
            }

            if let Some(sub_masks) = patch.get_mut("subMasks").and_then(|v| v.as_array_mut()) {
                hydrate_sub_masks(sub_masks, &mut cache);
            }
        }
    }

    if let Some(masks) = adjustments.get_mut("masks").and_then(|v| v.as_array_mut()) {
        for mask_container in masks {
            if let Some(sub_masks) = mask_container
                .get_mut("subMasks")
                .and_then(|v| v.as_array_mut())
            {
                hydrate_sub_masks(sub_masks, &mut cache);
            }
        }
    }
}

pub fn apply_all_transformations<'a, I: IntoCowImage<'a>>(
    image: I,
    adjustments: &serde_json::Value,
) -> (Cow<'a, DynamicImage>, (f32, f32)) {
    let start_time = std::time::Instant::now();
    let image = image.into_cow();
    let warped_image = apply_geometry_warp(image, adjustments);
    let blurred_image = crate::lens_blur::apply_lens_blur(warped_image, adjustments);

    let orientation_steps = adjustments["orientationSteps"].as_u64().unwrap_or(0) as u8;
    let rotation_degrees = adjustments["rotation"].as_f64().unwrap_or(0.0) as f32;
    let flip_horizontal = adjustments["flipHorizontal"].as_bool().unwrap_or(false);
    let flip_vertical = adjustments["flipVertical"].as_bool().unwrap_or(false);

    let coarse_rotated_image = apply_coarse_rotation(blurred_image, orientation_steps);
    let flipped_image = apply_flip(coarse_rotated_image, flip_horizontal, flip_vertical);
    let rotated_image = apply_rotation(flipped_image, rotation_degrees);

    let crop_data: Option<Crop> = serde_json::from_value(adjustments["crop"].clone()).ok();
    let crop_json = serde_json::to_value(crop_data).unwrap_or(serde_json::Value::Null);
    let cropped_image = apply_crop(rotated_image, &crop_json);

    let unscaled_crop_offset = crop_data.map_or((0.0, 0.0), |c| (c.x as f32, c.y as f32));

    let total_duration = start_time.elapsed();
    log::info!("apply_all_transformations took {:.2?}", total_duration);

    (cropped_image, unscaled_crop_offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GenericImageView, Rgb, RgbImage};

    fn coordinate_image(width: u32, height: u32) -> DynamicImage {
        let mut image = RgbImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                image.put_pixel(x, y, Rgb([x as u8 * 20, y as u8 * 30, 11]));
            }
        }
        DynamicImage::ImageRgb8(image)
    }

    #[test]
    fn transformation_pipeline_applies_source_pixel_crop_and_reports_offset() {
        let source = coordinate_image(6, 4);
        let adjustments = serde_json::json!({
            "crop": { "unit": "px", "x": 1.0, "y": 1.0, "width": 3.0, "height": 2.0 }
        });

        let (output, offset) = apply_all_transformations(&source, &adjustments);

        assert_eq!(output.dimensions(), (3, 2));
        assert_eq!(offset, (1.0, 1.0));
        assert_eq!(output.get_pixel(0, 0), source.get_pixel(1, 1));
        assert_eq!(output.get_pixel(2, 1), source.get_pixel(3, 2));
    }

    #[test]
    fn crop_coordinates_follow_coarse_rotation_and_flip() {
        let source = coordinate_image(4, 3);
        let adjustments = serde_json::json!({
            "orientationSteps": 1,
            "flipHorizontal": true,
            "crop": { "unit": "px", "x": 1.0, "y": 1.0, "width": 2.0, "height": 2.0 }
        });

        let (output, offset) = apply_all_transformations(&source, &adjustments);

        assert_eq!(output.dimensions(), (2, 2));
        assert_eq!(offset, (1.0, 1.0));
        let expected = source.rotate90().fliph().crop_imm(1, 1, 2, 2);
        assert_eq!(output.to_rgba8(), expected.to_rgba8());
    }
}
