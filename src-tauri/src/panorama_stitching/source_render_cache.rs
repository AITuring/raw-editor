//! Byte-bounded source reuse with a separate, run-local lossless disk budget.
use crate::panorama_utils::stack_pipeline::{
    cache_backing::{BackingFile, read_plane, write_plane},
    report::SourceRenderCacheReport,
};
use image::Rgb32FImage;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct SourceRenderCacheKey {
    pub(super) path: String,
    pub(super) sha256: [u8; 32],
    pub(super) dimensions: (u32, u32),
    pub(super) processing_fingerprint: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SourceRenderCacheStage {
    Render,
    QualityGate,
}
enum Payload {
    Memory(Arc<Rgb32FImage>),
    Disk {
        file: BackingFile,
        width: u32,
        height: u32,
    },
}
struct Entry {
    payload: Payload,
    bytes: usize,
}
pub(super) struct SourceRenderCache {
    entries: HashMap<SourceRenderCacheKey, Entry>,
    order: VecDeque<SourceRenderCacheKey>,
    stats: SourceRenderCacheReport,
}
impl SourceRenderCache {
    pub(super) fn new(max_bytes: usize) -> Self {
        Self::with_backing(max_bytes, 0)
    }
    pub(super) fn with_backing(max_bytes: usize, disk_bytes: usize) -> Self {
        let mut stats = SourceRenderCacheReport {
            max_bytes,
            ..Default::default()
        };
        stats.backing.max_bytes = if max_bytes == 0 { 0 } else { disk_bytes };
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            stats,
        }
    }
    pub(super) fn get(
        &mut self,
        key: &SourceRenderCacheKey,
        stage: SourceRenderCacheStage,
    ) -> Option<Arc<Rgb32FImage>> {
        let mut failed = false;
        let value = self
            .entries
            .get(key)
            .and_then(|entry| match &entry.payload {
                Payload::Memory(image) => Some(Arc::clone(image)),
                Payload::Disk {
                    file,
                    width,
                    height,
                } => {
                    let start = Instant::now();
                    let result =
                        file.read(|r| read_plane::<f32>(r, *width as usize * *height as usize * 3));
                    self.stats.backing.read_seconds += start.elapsed().as_secs_f64();
                    match result {
                        Ok(pixels) => {
                            self.stats.backing.hits += 1;
                            self.stats.backing.read_bytes += file.bytes;
                            Rgb32FImage::from_raw(*width, *height, pixels).map(Arc::new)
                        }
                        Err(_) => {
                            failed = true;
                            self.stats.backing.failures += 1;
                            None
                        }
                    }
                }
            });
        if failed {
            self.remove(key);
        }
        if value.is_some() {
            self.stats.hits += 1;
            match stage {
                SourceRenderCacheStage::Render => self.stats.render_hits += 1,
                SourceRenderCacheStage::QualityGate => self.stats.quality_gate_hits += 1,
            }
            self.order.retain(|k| k != key);
            self.order.push_back(key.clone());
        } else {
            self.stats.misses += 1;
            match stage {
                SourceRenderCacheStage::Render => self.stats.render_misses += 1,
                SourceRenderCacheStage::QualityGate => self.stats.quality_gate_misses += 1,
            }
        }
        value
    }
    pub(super) fn insert(&mut self, key: SourceRenderCacheKey, image: Arc<Rgb32FImage>) -> bool {
        // Include key copies and conservative hash/LRU/file bookkeeping in RAM.
        let metadata = 1024usize.saturating_add(key.path.capacity().saturating_mul(2));
        let pixels = image.as_raw().capacity().saturating_mul(4);
        let use_disk = pixels.saturating_add(metadata) > self.stats.max_bytes / 4
            && self.stats.backing.max_bytes > 0;
        let disk_bytes = if use_disk { image_bytes(&image) } else { 0 };
        let bytes = metadata.saturating_add(if use_disk { 0 } else { pixels });
        if bytes > self.stats.max_bytes || disk_bytes > self.stats.backing.max_bytes {
            self.stats.oversized_bypasses += 1;
            return false;
        }
        self.remove(&key);
        while self.stats.resident_bytes.saturating_add(bytes) > self.stats.max_bytes
            || self.stats.backing.resident_bytes.saturating_add(disk_bytes)
                > self.stats.backing.max_bytes
        {
            let Some(oldest) = self.order.front().cloned() else {
                break;
            };
            self.remove(&oldest);
            self.stats.evictions += 1;
            self.stats.backing.evictions += usize::from(use_disk);
        }
        let payload = if use_disk {
            let start = Instant::now();
            let result = BackingFile::write(disk_bytes, |w| write_plane(w, image.as_raw()));
            self.stats.backing.write_seconds += start.elapsed().as_secs_f64();
            let Ok(file) = result else {
                self.stats.backing.failures += 1;
                return false;
            };
            self.stats.backing.writes += 1;
            self.stats.backing.written_bytes += disk_bytes;
            self.stats.backing.resident_bytes += disk_bytes;
            self.stats.backing.peak_bytes = self
                .stats
                .backing
                .peak_bytes
                .max(self.stats.backing.resident_bytes);
            Payload::Disk {
                file,
                width: image.width(),
                height: image.height(),
            }
        } else {
            Payload::Memory(image)
        };
        self.stats.resident_bytes += bytes;
        self.stats.peak_bytes = self.stats.peak_bytes.max(self.stats.resident_bytes);
        self.order.push_back(key.clone());
        self.entries.insert(key, Entry { payload, bytes });
        self.stats.resident_entries = self.entries.len();
        true
    }
    fn remove(&mut self, key: &SourceRenderCacheKey) {
        if let Some(entry) = self.entries.remove(key) {
            self.stats.resident_bytes -= entry.bytes;
            if let Payload::Disk { file, .. } = entry.payload {
                self.stats.backing.resident_bytes -= file.bytes;
            }
        }
        self.order.retain(|k| k != key);
        self.entries.shrink_to_fit();
        self.order.shrink_to_fit();
        self.stats.resident_entries = self.entries.len();
    }
    pub(super) fn clear(&mut self) {
        self.entries.clear();
        self.entries.shrink_to_fit();
        self.order.clear();
        self.order.shrink_to_fit();
        self.stats.resident_bytes = 0;
        self.stats.resident_entries = 0;
        self.stats.backing.resident_bytes = 0;
    }
    pub(super) fn stats(&self) -> SourceRenderCacheReport {
        self.stats
    }
}

pub(super) const SOURCE_RENDER_CACHE_MAX_BYTES: usize = 512 * 1024 * 1024;

/// Allow an explicit acceptance/profile override while keeping the product
/// default conservative. The upper bound prevents an accidental environment
/// typo from turning this into an unbounded source-image cache.
pub(super) fn source_render_cache_max_bytes() -> usize {
    const HARD_MAX_BYTES: usize = 4 * 1024 * 1024 * 1024;
    std::env::var("RAW_EDITOR_STACK_SOURCE_RENDER_CACHE_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0 && value <= HARD_MAX_BYTES)
        .unwrap_or(SOURCE_RENDER_CACHE_MAX_BYTES)
}

/// Fingerprint the pixel-affecting settings that sit between SourcePreflight
/// and the post-normalisation image kept by this cache.  The fingerprint is
/// intentionally run-local: it only prevents an image prepared with another
/// exposure/gain policy from being reused in this run.
pub(super) fn processing_fingerprint(
    raw_exposure_reference: Option<f64>,
    shared_raw_display_gain: Option<f32>,
    fixed_gain_env: Option<&str>,
    skip_raw_normalization: bool,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    // Bump when the post-decode processing sequence changes.
    1u8.hash(&mut hasher);
    raw_exposure_reference.map(f64::to_bits).hash(&mut hasher);
    shared_raw_display_gain.map(f32::to_bits).hash(&mut hasher);
    fixed_gain_env.hash(&mut hasher);
    skip_raw_normalization.hash(&mut hasher);
    hasher.finish()
}

fn image_bytes(image: &Rgb32FImage) -> usize {
    (image.width() as usize)
        .saturating_mul(image.height() as usize)
        .saturating_mul(3)
        .saturating_mul(std::mem::size_of::<f32>())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(id: u8) -> SourceRenderCacheKey {
        SourceRenderCacheKey {
            path: format!("source-{id}.NEF"),
            sha256: [id; 32],
            dimensions: (2, 2),
            processing_fingerprint: 1,
        }
    }

    fn image(value: f32, width: u32, height: u32) -> Arc<Rgb32FImage> {
        Arc::new(Rgb32FImage::from_pixel(
            width,
            height,
            image::Rgb([value; 3]),
        ))
    }

    fn test_entry_bytes() -> usize {
        let mut probe = SourceRenderCache::new(usize::MAX);
        probe.insert(key(1), image(1.0, 2, 2));
        probe.stats().resident_bytes
    }

    #[test]
    fn hits_refresh_lru_and_keep_source_identity() {
        let mut cache = SourceRenderCache::new(test_entry_bytes() * 2);
        cache.insert(key(1), image(1.0, 2, 2));
        cache.insert(key(2), image(2.0, 2, 2));
        assert_eq!(
            cache
                .get(&key(1), SourceRenderCacheStage::Render)
                .expect("first source should hit")
                .get_pixel(0, 0)
                .0[0],
            1.0
        );
        cache.insert(key(3), image(3.0, 2, 2));
        assert!(cache.get(&key(1), SourceRenderCacheStage::Render).is_some());
        assert!(
            cache
                .get(&key(2), SourceRenderCacheStage::QualityGate)
                .is_none()
        );
        assert_eq!(cache.stats().hits, 2);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().render_hits, 2);
        assert_eq!(cache.stats().quality_gate_misses, 1);
    }

    #[test]
    fn byte_limit_evicts_oldest_and_bypasses_oversized_images() {
        let entry_bytes = test_entry_bytes();
        let mut cache = SourceRenderCache::new(entry_bytes);
        cache.insert(key(1), image(1.0, 2, 2));
        cache.insert(key(2), image(2.0, 2, 2));
        assert!(cache.get(&key(1), SourceRenderCacheStage::Render).is_none());
        assert!(cache.get(&key(2), SourceRenderCacheStage::Render).is_some());
        assert_eq!(cache.stats().evictions, 1);
        assert!(!cache.insert(key(3), image(3.0, 3, 3)));
        assert_eq!(cache.stats().oversized_bypasses, 1);
    }

    #[test]
    fn render_to_quality_gate_reuse_is_counted_by_stage() {
        let mut cache = SourceRenderCache::new(test_entry_bytes());
        cache.insert(key(1), image(1.0, 2, 2));

        assert!(
            cache
                .get(&key(1), SourceRenderCacheStage::QualityGate)
                .is_some()
        );
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.quality_gate_hits, 1);
        assert_eq!(stats.render_hits, 0);
    }

    #[test]
    fn oversized_source_backing_is_bit_exact_bounded_and_run_local() {
        let mut cache = SourceRenderCache::with_backing(4096, 2 * 32 * 32 * 12);
        let mut pixels = Rgb32FImage::new(32, 32);
        pixels.put_pixel(
            0,
            0,
            image::Rgb([-0.0, f32::from_bits(0x7fc01234), 0.12345679]),
        );
        let original = Arc::new(pixels);
        assert!(cache.insert(key(1), original.clone()));
        assert!(cache.insert(key(2), image(2.0, 32, 32)));
        let hit = cache
            .get(&key(1), SourceRenderCacheStage::QualityGate)
            .unwrap();
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(hit.as_raw()),
            bytemuck::cast_slice::<_, u8>(original.as_raw())
        );
        assert!(cache.insert(key(3), image(3.0, 32, 32)));
        assert!(cache.get(&key(2), SourceRenderCacheStage::Render).is_none());
        assert!(cache.stats().peak_bytes <= 4096);
        assert!(cache.stats().backing.peak_bytes <= 2 * 32 * 32 * 12);
        let mut changed = key(1);
        changed.sha256 = [99; 32];
        assert!(
            cache
                .get(&changed, SourceRenderCacheStage::Render)
                .is_none()
        );
        let Payload::Disk { file, .. } = &cache.entries[&key(1)].payload else {
            panic!("expected backing")
        };
        file.corrupt();
        assert!(
            cache
                .get(&key(1), SourceRenderCacheStage::QualityGate)
                .is_none()
        );
        assert_eq!(cache.stats().backing.failures, 1);
        cache.clear();
        assert_eq!(cache.stats().resident_bytes, 0);
        assert_eq!(cache.stats().backing.resident_bytes, 0);
        let mut tiny = SourceRenderCache::with_backing(4096, 1);
        assert!(!tiny.insert(key(1), original));
        assert_eq!(tiny.stats().backing.resident_bytes, 0);
    }

    #[test]
    fn processing_fingerprint_separates_pixel_affecting_settings() {
        let baseline = processing_fingerprint(Some(4.0), Some(1.1), None, false);
        assert_ne!(
            baseline,
            processing_fingerprint(Some(5.0), Some(1.1), None, false)
        );
        assert_ne!(
            baseline,
            processing_fingerprint(Some(4.0), Some(1.2), None, false)
        );
        assert_ne!(
            baseline,
            processing_fingerprint(Some(4.0), Some(1.1), Some("1.4"), false)
        );
        assert_ne!(
            baseline,
            processing_fingerprint(Some(4.0), Some(1.1), None, true)
        );
    }
}
