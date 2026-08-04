use crate::AppState;
use image::DynamicImage;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

const MIB: usize = 1024 * 1024;

#[cfg(target_os = "android")]
pub const DECODED_IMAGE_CACHE_MAX_BYTES: usize = 384 * MIB;
#[cfg(not(target_os = "android"))]
pub const DECODED_IMAGE_CACHE_MAX_BYTES: usize = 1024 * MIB;

pub const LUT_CACHE_MAX_ENTRIES: usize = 8;
pub const LUT_CACHE_MAX_BYTES: usize = 64 * MIB;
pub const MASK_CACHE_MAX_ENTRIES: usize = 32;
pub const MASK_CACHE_MAX_BYTES: usize = 128 * MIB;
pub const GEOMETRY_CACHE_MAX_ENTRIES: usize = 6;
pub const GEOMETRY_CACHE_MAX_BYTES: usize = 96 * MIB;
pub const THUMBNAIL_GEOMETRY_CACHE_MAX_ENTRIES: usize = 30;
pub const THUMBNAIL_GEOMETRY_CACHE_MAX_BYTES: usize = 128 * MIB;

/// A small, dependency-free LRU cache with both entry-count and memory budgets.
///
/// Image editing entries can differ by hundreds of megabytes, so a count-only
/// limit is not sufficient for predictable memory use. Values larger than the
/// byte budget are deliberately not cached; callers still retain and use the
/// value they just produced.
pub struct BudgetedCache<K, V> {
    max_entries: usize,
    max_bytes: usize,
    current_bytes: usize,
    items: Vec<(K, V, usize)>,
}

impl<K: Eq, V> BudgetedCache<K, V> {
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            max_entries,
            max_bytes,
            current_bytes: 0,
            items: Vec::with_capacity(max_entries),
        }
    }

    pub fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let position = self
            .items
            .iter()
            .position(|(stored_key, _, _)| stored_key.borrow() == key)?;
        let item = self.items.remove(position);
        self.items.push(item);
        self.items.last().map(|(_, value, _)| value)
    }

    pub fn insert(&mut self, key: K, value: V, weight_bytes: usize) -> bool {
        if let Some(position) = self
            .items
            .iter()
            .position(|(stored_key, _, _)| stored_key == &key)
        {
            let (_, _, old_weight) = self.items.remove(position);
            self.current_bytes = self.current_bytes.saturating_sub(old_weight);
        }

        let weight_bytes = weight_bytes.max(1);
        if self.max_entries == 0 || weight_bytes > self.max_bytes {
            return false;
        }

        while !self.items.is_empty()
            && (self.items.len() >= self.max_entries
                || self.current_bytes.saturating_add(weight_bytes) > self.max_bytes)
        {
            let (_, _, evicted_weight) = self.items.remove(0);
            self.current_bytes = self.current_bytes.saturating_sub(evicted_weight);
        }

        self.current_bytes = self.current_bytes.saturating_add(weight_bytes);
        self.items.push((key, value, weight_bytes));
        true
    }

    pub fn clear(&mut self) {
        self.items.clear();
        self.current_bytes = 0;
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.items.len()
    }

    #[cfg(test)]
    fn current_bytes(&self) -> usize {
        self.current_bytes
    }
}

pub fn dynamic_image_weight(image: &DynamicImage) -> usize {
    image.as_bytes().len().max(1)
}

pub const GEOMETRY_KEYS: &[&str] = &[
    "transformDistortion",
    "transformVertical",
    "transformHorizontal",
    "transformRotate",
    "transformAspect",
    "transformScale",
    "transformXOffset",
    "transformYOffset",
    "lensDistortionAmount",
    "lensVignetteAmount",
    "lensTcaAmount",
    "lensDistortionParams",
    "lensMaker",
    "lensModel",
    "lensDistortionEnabled",
    "lensTcaEnabled",
    "lensVignetteEnabled",
];

pub fn calculate_thumbnail_base_hash(adjustments: &serde_json::Value) -> u64 {
    let mut hasher = DefaultHasher::new();

    calculate_geometry_hash(adjustments).hash(&mut hasher);

    let effects_visible = adjustments
        .get("sectionVisibility")
        .and_then(|v| v.get("effects"))
        .and_then(|s| s.as_bool())
        .unwrap_or(true);

    let blur_enabled = effects_visible && adjustments["lensBlurEnabled"].as_bool().unwrap_or(false);
    blur_enabled.hash(&mut hasher);

    if blur_enabled {
        let blur_keys = [
            "lensBlurAmount",
            "lensBlurDiffusion",
            "lensBlurShape",
            "lensBlurMinDepth",
            "lensBlurMaxDepth",
            "lensBlurMinFade",
            "lensBlurMaxFade",
            "lensBlurDepthMap",
        ];

        for key in blur_keys {
            if let Some(val) = adjustments.get(key) {
                key.hash(&mut hasher);
                val.to_string().hash(&mut hasher);
            }
        }
    }

    hasher.finish()
}

pub fn calculate_geometry_hash(adjustments: &serde_json::Value) -> u64 {
    let mut hasher = DefaultHasher::new();

    if let Some(patches) = adjustments.get("aiPatches") {
        patches.to_string().hash(&mut hasher);
    }

    adjustments["orientationSteps"].as_u64().hash(&mut hasher);

    for key in GEOMETRY_KEYS {
        if let Some(val) = adjustments.get(key) {
            key.hash(&mut hasher);
            val.to_string().hash(&mut hasher);
        }
    }

    hasher.finish()
}

pub fn calculate_visual_hash(path: &str, adjustments: &serde_json::Value) -> u64 {
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);

    if let Some(obj) = adjustments.as_object() {
        for (key, value) in obj {
            if GEOMETRY_KEYS.contains(&key.as_str()) {
                continue;
            }

            match key.as_str() {
                "crop" | "rotation" | "orientationSteps" | "flipHorizontal" | "flipVertical" => (),
                _ => {
                    key.hash(&mut hasher);
                    value.to_string().hash(&mut hasher);
                }
            }
        }
    }

    hasher.finish()
}

pub fn calculate_transform_hash(adjustments: &serde_json::Value) -> u64 {
    let mut hasher = DefaultHasher::new();

    let orientation_steps = adjustments["orientationSteps"].as_u64().unwrap_or(0);
    orientation_steps.hash(&mut hasher);

    let rotation = adjustments["rotation"].as_f64().unwrap_or(0.0);
    (rotation.to_bits()).hash(&mut hasher);

    let flip_h = adjustments["flipHorizontal"].as_bool().unwrap_or(false);
    flip_h.hash(&mut hasher);

    let flip_v = adjustments["flipVertical"].as_bool().unwrap_or(false);
    flip_v.hash(&mut hasher);

    let effects_visible = adjustments
        .get("sectionVisibility")
        .and_then(|v| v.get("effects"))
        .and_then(|s| s.as_bool())
        .unwrap_or(true);

    let blur_enabled = effects_visible && adjustments["lensBlurEnabled"].as_bool().unwrap_or(false);
    blur_enabled.hash(&mut hasher);
    if blur_enabled {
        if let Some(val) = adjustments.get("lensBlurAmount") {
            val.to_string().hash(&mut hasher);
        }
        if let Some(val) = adjustments.get("lensBlurDiffusion") {
            val.to_string().hash(&mut hasher);
        }
        if let Some(val) = adjustments.get("lensBlurShape") {
            val.as_str().unwrap_or("").hash(&mut hasher);
        }
        if let Some(val) = adjustments.get("lensBlurMinDepth") {
            val.to_string().hash(&mut hasher);
        }
        if let Some(val) = adjustments.get("lensBlurMaxDepth") {
            val.to_string().hash(&mut hasher);
        }
        if let Some(val) = adjustments.get("lensBlurMinFade") {
            val.to_string().hash(&mut hasher);
        }
        if let Some(val) = adjustments.get("lensBlurMaxFade") {
            val.to_string().hash(&mut hasher);
        }
        if let Some(val) = adjustments.get("lensBlurDepthMap") {
            val.as_str().unwrap_or("").len().hash(&mut hasher);
        }
    }

    if let Some(crop_val) = adjustments.get("crop")
        && !crop_val.is_null()
    {
        crop_val.to_string().hash(&mut hasher);
    }

    for key in GEOMETRY_KEYS {
        if let Some(val) = adjustments.get(key) {
            key.hash(&mut hasher);
            val.to_string().hash(&mut hasher);
        }
    }

    if let Some(patches_val) = adjustments.get("aiPatches")
        && let Some(patches_arr) = patches_val.as_array()
    {
        patches_arr.len().hash(&mut hasher);

        for patch in patches_arr {
            if let Some(id) = patch.get("id").and_then(|v| v.as_str()) {
                id.hash(&mut hasher);
            }

            let is_visible = patch
                .get("visible")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            is_visible.hash(&mut hasher);

            if let Some(patch_data) = patch.get("patchData") {
                let color_len = patch_data
                    .get("color")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .len();
                color_len.hash(&mut hasher);

                let mask_len = patch_data
                    .get("mask")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .len();
                mask_len.hash(&mut hasher);
            } else {
                let data_len = patch
                    .get("patchDataBase64")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .len();
                data_len.hash(&mut hasher);
            }

            if let Some(sub_masks_val) = patch.get("subMasks") {
                sub_masks_val.to_string().hash(&mut hasher);
            }

            let invert = patch
                .get("invert")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            invert.hash(&mut hasher);
        }
    }

    hasher.finish()
}

pub fn calculate_full_job_hash(path: &str, adjustments: &serde_json::Value) -> u64 {
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    adjustments.to_string().hash(&mut hasher);
    hasher.finish()
}

struct DecodedImageCacheEntry {
    path: String,
    image: Arc<DynamicImage>,
    exif: HashMap<String, String>,
    weight: usize,
}

pub struct DecodedImageCache {
    capacity: usize,
    max_bytes: usize,
    current_bytes: usize,
    items: Vec<DecodedImageCacheEntry>,
}

impl DecodedImageCache {
    pub fn new(capacity: usize, max_bytes: usize) -> Self {
        Self {
            capacity,
            max_bytes,
            current_bytes: 0,
            items: Vec::with_capacity(capacity),
        }
    }

    pub fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity;
        while self.items.len() > self.capacity {
            let entry = self.items.remove(0);
            self.current_bytes = self.current_bytes.saturating_sub(entry.weight);
        }
    }

    pub fn get(&mut self, path: &str) -> Option<(Arc<DynamicImage>, HashMap<String, String>)> {
        if let Some(pos) = self.items.iter().position(|entry| entry.path == path) {
            let entry = self.items.remove(pos);
            let result = (entry.image.clone(), entry.exif.clone());
            self.items.push(entry);
            Some(result)
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        self.items.clear();
        self.current_bytes = 0;
    }

    pub fn insert(
        &mut self,
        path: String,
        image: Arc<DynamicImage>,
        exif: HashMap<String, String>,
    ) {
        if let Some(pos) = self.items.iter().position(|entry| entry.path == path) {
            let entry = self.items.remove(pos);
            self.current_bytes = self.current_bytes.saturating_sub(entry.weight);
        }

        let exif_weight = exif.iter().fold(0usize, |total, (key, value)| {
            total.saturating_add(key.len()).saturating_add(value.len())
        });
        let weight = dynamic_image_weight(&image).saturating_add(exif_weight);

        if self.capacity == 0 || weight > self.max_bytes {
            return;
        }

        while !self.items.is_empty()
            && (self.items.len() >= self.capacity
                || self.current_bytes.saturating_add(weight) > self.max_bytes)
        {
            let entry = self.items.remove(0);
            self.current_bytes = self.current_bytes.saturating_sub(entry.weight);
        }

        self.current_bytes = self.current_bytes.saturating_add(weight);
        self.items.push(DecodedImageCacheEntry {
            path,
            image,
            exif,
            weight,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::{BudgetedCache, DecodedImageCache};
    use image::DynamicImage;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn budgeted_cache_evicts_least_recently_used_entry() {
        let mut cache = BudgetedCache::new(2, 100);
        assert!(cache.insert("first", 1, 10));
        assert!(cache.insert("second", 2, 10));
        assert_eq!(cache.get(&"first"), Some(&1));

        assert!(cache.insert("third", 3, 10));

        assert!(cache.get(&"second").is_none());
        assert_eq!(cache.get(&"first"), Some(&1));
        assert_eq!(cache.get(&"third"), Some(&3));
    }

    #[test]
    fn budgeted_cache_enforces_byte_budget_and_replacement_weight() {
        let mut cache = BudgetedCache::new(4, 10);
        assert!(cache.insert("first", 1, 6));
        assert!(cache.insert("second", 2, 4));
        assert_eq!(cache.current_bytes(), 10);

        assert!(cache.insert("second", 20, 7));

        assert!(cache.get(&"first").is_none());
        assert_eq!(cache.get(&"second"), Some(&20));
        assert_eq!(cache.current_bytes(), 7);
    }

    #[test]
    fn budgeted_cache_does_not_retain_oversized_entries() {
        let mut cache = BudgetedCache::new(4, 10);
        assert!(!cache.insert("oversized", 1, 11));
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.current_bytes(), 0);
    }

    #[test]
    fn decoded_image_cache_uses_memory_budget_in_addition_to_count() {
        let image = Arc::new(DynamicImage::new_rgb8(4, 4));
        let image_bytes = image.as_bytes().len();
        let mut cache = DecodedImageCache::new(3, image_bytes + 1);

        cache.insert("first".to_string(), image.clone(), HashMap::new());
        cache.insert("second".to_string(), image, HashMap::new());

        assert!(cache.get("first").is_none());
        assert!(cache.get("second").is_some());
    }

    #[test]
    fn decoded_image_cache_skips_an_image_larger_than_its_budget() {
        let image = Arc::new(DynamicImage::new_rgb8(4, 4));
        let mut cache = DecodedImageCache::new(3, image.as_bytes().len() - 1);

        cache.insert("oversized".to_string(), image, HashMap::new());

        assert!(cache.get("oversized").is_none());
    }
}

#[tauri::command]
pub fn clear_image_caches(state: tauri::State<AppState>) {
    if let Ok(mut decoded_cache) = state.decoded_image_cache.lock() {
        decoded_cache.clear();
    }
    if let Ok(mut gpu_cache) = state.gpu_image_cache.lock() {
        *gpu_cache = None;
    }
    if let Ok(mut preview_cache) = state.cached_preview.lock() {
        *preview_cache = None;
    }
    if let Ok(mut warped_cache) = state.full_warped_cache.lock() {
        *warped_cache = None;
    }
    if let Ok(mut transformed_cache) = state.full_transformed_cache.lock() {
        *transformed_cache = None;
    }
}

#[tauri::command]
pub fn clear_session_caches(state: tauri::State<AppState>) {
    if let Ok(mut patch_cache) = state.patch_cache.lock() {
        patch_cache.clear();
    }
    if let Ok(mut mask_cache) = state.mask_cache.lock() {
        mask_cache.clear();
    }
    if let Ok(mut geometry_cache) = state.geometry_cache.lock() {
        geometry_cache.clear();
    }
}
