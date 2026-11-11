//! Run-scoped Source_RAW preflight.
//!
//! A focus-stack run used to reopen the same source for the cache key, focal
//! length detection, exposure reference, preparation, rendering and Quality
//! Gate.  This module performs the read-only file and metadata work once and
//! keeps the mapped bytes available to the later stages.  Decoded full-size
//! images are deliberately not retained here; the existing bounded-memory
//! policy still controls that part of the pipeline.

use crate::file_management::{parse_virtual_path, read_file_mapped};
use crate::formats::is_raw_file;
use image::ImageReader;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::source_prep::stack_raw_exposure_from_bytes;

fn source_real_path(path: &str) -> PathBuf {
    if path.rsplit_once("?vc=").is_some() {
        parse_virtual_path(path).0
    } else {
        PathBuf::from(path)
    }
}

#[derive(Default)]
pub(super) struct SourcePreflightCounters {
    pub(super) stat_count: AtomicUsize,
    pub(super) file_map_count: AtomicUsize,
    pub(super) hash_count: AtomicUsize,
    pub(super) exif_parse_count: AtomicUsize,
    pub(super) preflight_reuse_count: AtomicUsize,
    pub(super) raw_decode_count: AtomicUsize,
    pub(super) failed_count: AtomicUsize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct SourcePreflightStats {
    pub(super) source_count: usize,
    pub(super) stat_count: usize,
    pub(super) file_map_count: usize,
    pub(super) hash_count: usize,
    pub(super) exif_parse_count: usize,
    pub(super) preflight_reuse_count: usize,
    pub(super) raw_decode_count: usize,
    pub(super) failed_count: usize,
}

impl SourcePreflightCounters {
    pub(super) fn snapshot(&self, source_count: usize) -> SourcePreflightStats {
        SourcePreflightStats {
            source_count,
            stat_count: self.stat_count.load(Ordering::Relaxed),
            file_map_count: self.file_map_count.load(Ordering::Relaxed),
            hash_count: self.hash_count.load(Ordering::Relaxed),
            exif_parse_count: self.exif_parse_count.load(Ordering::Relaxed),
            preflight_reuse_count: self.preflight_reuse_count.load(Ordering::Relaxed),
            raw_decode_count: self.raw_decode_count.load(Ordering::Relaxed),
            failed_count: self.failed_count.load(Ordering::Relaxed),
        }
    }
}

pub(super) struct SourcePreflightRecord {
    pub(super) path: String,
    pub(super) bytes: Arc<memmap2::Mmap>,
    pub(super) dimensions: Option<(u32, u32)>,
    pub(super) sha256: [u8; 32],
    pub(super) focal_length_35mm: Option<f64>,
    pub(super) exif: HashMap<String, String>,
    pub(super) raw_exposure: Option<f64>,
    pub(super) counters: Arc<SourcePreflightCounters>,
}

pub(super) struct SourcePreflight {
    paths: Vec<String>,
    entries: Vec<Option<SourcePreflightRecord>>,
    by_path: HashMap<String, usize>,
    counters: Arc<SourcePreflightCounters>,
}

impl SourcePreflight {
    pub(super) fn collect(paths: &[String]) -> Self {
        let counters = Arc::new(SourcePreflightCounters::default());
        let mut seen = HashSet::with_capacity(paths.len());
        let unique_paths = paths
            .iter()
            .filter(|path| seen.insert((*path).clone()))
            .cloned()
            .collect::<Vec<_>>();
        let entries: Vec<Option<SourcePreflightRecord>> = unique_paths
            .par_iter()
            .map(|path| {
                let real_path = source_real_path(path);
                counters.stat_count.fetch_add(1, Ordering::Relaxed);
                let mapped = match read_file_mapped(&real_path) {
                    Ok(mapped) => mapped,
                    Err(error) => {
                        counters.failed_count.fetch_add(1, Ordering::Relaxed);
                        println!("Focus stack preflight: failed to read {path}: {error}");
                        return None;
                    }
                };
                counters.file_map_count.fetch_add(1, Ordering::Relaxed);
                let bytes = Arc::new(mapped);
                let sha256 = Sha256::digest(&bytes[..]).into();
                counters.hash_count.fetch_add(1, Ordering::Relaxed);
                let exif = crate::exif_processing::read_exif_data_from_bytes(path, &bytes);
                counters.exif_parse_count.fetch_add(1, Ordering::Relaxed);
                let dimensions = ImageReader::new(Cursor::new(&bytes[..]))
                    .with_guessed_format()
                    .ok()
                    .and_then(|reader| reader.into_dimensions().ok());
                let focal_length_35mm =
                    crate::exif_processing::focal_length_35mm_from_bytes(&bytes);
                let raw_exposure = is_raw_file(path)
                    .then(|| stack_raw_exposure_from_bytes(&bytes))
                    .flatten();
                Some(SourcePreflightRecord {
                    path: path.clone(),
                    bytes,
                    dimensions,
                    sha256,
                    focal_length_35mm,
                    exif,
                    raw_exposure,
                    counters: Arc::clone(&counters),
                })
            })
            .collect();
        let by_path = entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| entry.as_ref().map(|entry| (entry.path.clone(), index)))
            .collect();
        Self {
            paths: paths.to_vec(),
            entries,
            by_path,
            counters,
        }
    }

    pub(super) fn record(&self, path: &str) -> Option<&SourcePreflightRecord> {
        self.by_path
            .get(path)
            .and_then(|&index| self.entries.get(index))
            .and_then(Option::as_ref)
    }

    pub(super) fn sha256_before(&self) -> Vec<Option<[u8; 32]>> {
        self.paths
            .iter()
            .map(|path| self.record(path).map(|entry| entry.sha256))
            .collect()
    }

    pub(super) fn stats(&self) -> SourcePreflightStats {
        self.counters.snapshot(self.paths.len())
    }

    pub(super) fn requires_bounded_memory(&self) -> bool {
        if self.paths.len() > super::SCALABLE_STACK_THRESHOLD {
            return true;
        }
        let mut pixels = 0u64;
        for (path, entry) in self.paths.iter().zip(&self.entries) {
            // RAW decoders expand the sensor plane before the dimensions
            // reported by the generic image reader become useful. Preserve
            // the historical safety rule: every RAW selection stays on the
            // bounded preparation path even when preflight could inspect a
            // container-level width/height.
            if is_raw_file(path) {
                return true;
            }
            let Some((width, height)) = entry.as_ref().and_then(|entry| entry.dimensions) else {
                return true;
            };
            pixels = pixels.saturating_add(u64::from(width) * u64::from(height));
            if pixels > super::MAX_RETAINED_STACK_PIXELS {
                return true;
            }
        }
        false
    }

    pub(super) fn has_mixed_focal_lengths(&self) -> bool {
        super::source_prep::focal_lengths_span_multiple_lenses(
            self.entries
                .iter()
                .filter_map(|entry| entry.as_ref().and_then(|entry| entry.focal_length_35mm)),
        )
    }

    /// Per-camera median RAW exposure (time × ISO / aperture²), matching
    /// `source_prep::stack_raw_exposure_reference` without re-reading sources.
    pub(super) fn raw_exposure_reference(
        &self,
    ) -> Option<super::source_prep::RawExposureReference> {
        let mut samples = Vec::with_capacity(self.entries.len());
        for path in &self.paths {
            if !is_raw_file(path) {
                return None;
            }
            let entry = self.record(path)?;
            let camera = (
                entry.exif.get("Make")?.clone(),
                entry.exif.get("Model")?.clone(),
            );
            samples.push((camera, entry.raw_exposure?));
        }
        super::source_prep::RawExposureReference::from_samples(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn collects_each_source_once_and_reuses_the_mapped_record() {
        let directory = tempfile::tempdir().expect("temporary source directory");
        let path = directory.path().join("source.png");
        let image = RgbImage::from_pixel(3, 2, Rgb([12, 34, 56]));
        image.save(&path).expect("fixture image must be writable");
        let path = path.to_string_lossy().into_owned();
        let paths = vec![path.clone(), path.clone()];

        let preflight = SourcePreflight::collect(&paths);
        let stats = preflight.stats();
        assert_eq!(stats.source_count, 2);
        assert_eq!(stats.stat_count, 1);
        assert_eq!(stats.file_map_count, 1);
        assert_eq!(stats.hash_count, 1);
        assert_eq!(stats.exif_parse_count, 1);
        assert_eq!(stats.failed_count, 0);
        assert_eq!(preflight.sha256_before().len(), 2);
        assert!(preflight.sha256_before().iter().all(Option::is_some));
        assert_eq!(
            preflight.record(&path).and_then(|record| record.dimensions),
            Some((3, 2))
        );
        assert!(!preflight.requires_bounded_memory());
    }

    #[test]
    fn raw_sources_always_use_bounded_preparation_even_when_dimensions_are_known() {
        let directory = tempfile::tempdir().expect("temporary source directory");
        let path = directory.path().join("source.nef");
        let fixture_path = directory.path().join("source.png");
        let image = RgbImage::from_pixel(3, 2, Rgb([12, 34, 56]));
        image
            .save(&fixture_path)
            .expect("fixture image must be writable");
        std::fs::rename(&fixture_path, &path).expect("RAW fixture must be renamed");
        let path = path.to_string_lossy().into_owned();

        let preflight = SourcePreflight::collect(std::slice::from_ref(&path));
        assert_eq!(
            preflight.record(&path).and_then(|record| record.dimensions),
            Some((3, 2))
        );
        assert!(preflight.requires_bounded_memory());
    }
}
