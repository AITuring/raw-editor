//! Transformed preview images and the full-resolution warp cache.
//!
//! The transformed full-resolution image is cached per transform hash and shares
//! the decoded original when no transform applies.

use crate::adjustment_utils::apply_all_transformations;
use crate::app_state::AppState;
use crate::app_state::LoadedImage;
use crate::cache_utils::calculate_geometry_hash;
use crate::cache_utils::calculate_transform_hash;
use crate::image_loader::composite_patches_on_image;
use crate::image_processing::apply_cpu_default_raw_processing;
use crate::image_processing::apply_geometry_warp;
use crate::image_processing::downscale_f32_image;
use image::DynamicImage;
use image::GenericImageView;
use std::borrow::Cow;
use std::sync::Arc;

use super::get_original_image;

pub type TransformedPreview = (Arc<DynamicImage>, f32, (f32, f32));

pub fn generate_transformed_preview(
    state: &tauri::State<AppState>,
    loaded_image: &LoadedImage,
    adjustments: &serde_json::Value,
    preview_dim: u32,
) -> Result<TransformedPreview, String> {
    let transform_hash = calculate_transform_hash(adjustments);

    let (transformed_full_res, unscaled_crop_offset) = {
        let mut cache_lock = state.full_transformed_cache.lock().unwrap();
        if let Some((hash, img, offset)) = cache_lock.as_ref() {
            if *hash == transform_hash {
                (Arc::clone(img), *offset)
            } else {
                let (arc_img, offset) = compute_full_transformed_res(loaded_image, adjustments)?;
                *cache_lock = Some((transform_hash, Arc::clone(&arc_img), offset));
                (arc_img, offset)
            }
        } else {
            let (arc_img, offset) = compute_full_transformed_res(loaded_image, adjustments)?;
            *cache_lock = Some((transform_hash, Arc::clone(&arc_img), offset));
            (arc_img, offset)
        }
    };

    let (full_res_w, full_res_h) = transformed_full_res.dimensions();

    let final_preview_base = if full_res_w > preview_dim || full_res_h > preview_dim {
        Arc::new(downscale_f32_image(
            &transformed_full_res,
            preview_dim,
            preview_dim,
        ))
    } else {
        Arc::clone(&transformed_full_res)
    };

    let scale_for_gpu = if full_res_w > 0 {
        final_preview_base.width() as f32 / full_res_w as f32
    } else {
        1.0
    };

    Ok((final_preview_base, scale_for_gpu, unscaled_crop_offset))
}

pub(super) fn compute_full_transformed_res(
    loaded_image: &LoadedImage,
    adjustments: &serde_json::Value,
) -> Result<(Arc<DynamicImage>, (f32, f32)), String> {
    let has_patches = adjustments
        .get("aiPatches")
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty());
    let patched_original_image = if has_patches {
        Cow::Owned(
            composite_patches_on_image(&loaded_image.image, adjustments)
                .map_err(|e| format!("Failed to composite AI patches: {}", e))?,
        )
    } else {
        Cow::Borrowed(loaded_image.image.as_ref())
    };

    let (transformed_img, offset) = apply_all_transformations(patched_original_image, adjustments);
    Ok((
        transformed_image_into_arc(transformed_img, &loaded_image.image),
        offset,
    ))
}

pub(super) fn transformed_image_into_arc<'a>(
    image: Cow<'a, DynamicImage>,
    original: &'a Arc<DynamicImage>,
) -> Arc<DynamicImage> {
    match image {
        Cow::Borrowed(borrowed) if std::ptr::eq(borrowed, original.as_ref()) => {
            Arc::clone(original)
        }
        Cow::Borrowed(borrowed) => Arc::new(borrowed.clone()),
        Cow::Owned(owned) => Arc::new(owned),
    }
}

pub fn get_cached_full_warped_image(
    state: &tauri::State<AppState>,
    js_adjustments: &serde_json::Value,
) -> Result<Arc<DynamicImage>, String> {
    let geo_hash = calculate_geometry_hash(js_adjustments);

    let cached_warped = {
        let cache_lock = state.full_warped_cache.lock().unwrap();
        if let Some((hash, img)) = cache_lock.as_ref()
            && *hash == geo_hash
        {
            Some(Arc::clone(img))
        } else {
            None
        }
    };
    if let Some(cached_warped) = cached_warped {
        if let Ok(mut range_source_cache) = state.range_mask_source_cache.lock() {
            *range_source_cache = None;
        }
        return Ok(cached_warped);
    }

    let (base_arc, is_raw) = get_original_image(state)?;
    let cached_range_source = if is_raw {
        state
            .range_mask_source_cache
            .lock()
            .ok()
            .and_then(|mut cache| cache.take())
            .and_then(|(cached_base, cached_source)| {
                Arc::ptr_eq(&cached_base, &base_arc).then_some(cached_source)
            })
    } else {
        None
    };
    let mut cow_image = cached_range_source
        .as_deref()
        .map_or_else(|| Cow::Borrowed(base_arc.as_ref()), Cow::Borrowed);

    if is_raw && cached_range_source.is_none() {
        apply_cpu_default_raw_processing(cow_image.to_mut());
    }

    let warped_image = apply_geometry_warp(cow_image, js_adjustments).into_owned();
    let warped_arc = Arc::new(warped_image);

    {
        let mut cache_lock = state.full_warped_cache.lock().unwrap();
        *cache_lock = Some((geo_hash, Arc::clone(&warped_arc)));
    }
    if let Ok(mut range_source_cache) = state.range_mask_source_cache.lock() {
        *range_source_cache = None;
    }

    Ok(warped_arc)
}
