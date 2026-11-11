//! Identity of a run-local native station fusion, including its source evidence.
//!
//! Immutable source pixels/metadata are fingerprinted once per station. A
//! rendering-prior repair changes the dynamic geometry key, so it cannot reuse
//! masks or source sampling made for the previous station coordinate system.

use crate::panorama_utils::shared::{Feature, FocusLayerWarp, ImageInfo};
use crate::panorama_utils::stitching::Projection;
use image::GrayImage;
use nalgebra::Matrix3;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct NativeFusionKey(pub(super) [u8; 32]);

/// Snapshot only after the run's source preparation and grouping are complete.
/// The borrowed `ImageInfo`s must stay immutable while this snapshot is used.
/// Rebuild it if a member, its metadata, prepared pixels, or preflight identity
/// changes. The pipeline's group slices hold immutable source borrows for this
/// whole phase; geometric repairs are passed separately to `cache_key`.
pub(super) struct NativeFusionSourceKey([u8; 32]);

impl NativeFusionSourceKey {
    pub(super) fn new(group: &[&ImageInfo], source_identity_fingerprint: [u8; 32]) -> Self {
        let mut hash = Encoder::new(b"native-fusion-sources-v1");
        hash.bytes(&source_identity_fingerprint);
        hash.len(group.len());
        for image in group {
            // Include the complete source record, not pointer identity: owner
            // order, source support and registration evidence affect fusion.
            hash.len(image.id);
            hash.bytes(image.filename.as_bytes());
            hash.u32(image.width);
            hash.u32(image.height);
            hash.gray(&image.alignment_image);
            hash.flag(image.full_image.is_some());
            if let Some(pixels) = image.full_image.as_ref() {
                hash.u32(pixels.width());
                hash.u32(pixels.height());
                hash.len(pixels.as_raw().len());
                // Keep this one-time snapshot linear in bandwidth, without a
                // separate SHA update for every channel of a full RAW image.
                if cfg!(target_endian = "little") {
                    hash.0
                        .update(bytemuck::cast_slice::<f32, u8>(pixels.as_raw()));
                } else {
                    for value in pixels.as_raw() {
                        hash.u32(value.to_bits());
                    }
                }
            }
            hash.f64(image.scale_factor);
            hash.option_f64(image.focal_length_35mm);
            // Calibration's scalar fields are private to lens_correction.
            // Its derived Debug contains every scalar in a fixed field order
            // and uses round-trip float formatting (no unordered containers).
            hash.bytes(format!("{:?}", image.source_footprint).as_bytes());
            hash.flag(image.overview_reference);
            hash.features(&image.features);
            hash.features(&image.top_features);
            hash.flag(image.foreground_range.is_some());
            if let Some((lower, upper)) = image.foreground_range {
                hash.f64(lower);
                hash.f64(upper);
            }
            hash.flag(image.foreground_mask.is_some());
            if let Some(mask) = image.foreground_mask.as_ref() {
                hash.gray(mask);
            }
            for values in [&image.horizontal_edge_rows, &image.vertical_edge_columns] {
                hash.len(values.len());
                for &value in values {
                    hash.f64(value);
                }
            }
        }
        Self(hash.finish())
    }

    pub(super) fn cache_key(
        &self,
        homographies: &HashMap<usize, Matrix3<f64>>,
        projection: Projection,
        focus_warp: Option<&FocusLayerWarp>,
        station_index: usize,
        processing_fingerprint: u64,
    ) -> NativeFusionKey {
        let mut hash = Encoder::new(b"native-fusion-result-v1");
        hash.bytes(&self.0);
        hash.len(station_index);
        hash.u64(processing_fingerprint);
        hash.u32(match projection {
            Projection::Planar => 0,
            Projection::Cylindrical => 1,
            Projection::Spherical => 2,
        });
        // Conservatively include the complete station-local map, including
        // missing/extra entries, and canonicalize its iteration order.
        hash.matrices(homographies);
        hash.flag(focus_warp.is_some());
        if let Some(warp) = focus_warp {
            hash.len(warp.bands.len());
            // Band ordering is meaningful to sampling, unlike map ordering.
            for band in &warp.bands {
                hash.matrices(&band.homographies);
                hash.ranges(&band.source_ranges);
                hash.ranges(&band.source_x_ranges);
                hash.flag(band.relax_foreground_seam);
                hash.flag(band.foreground_only);
                hash.flag(band.physical_edge);
            }
        }
        // These run options are consumed inside native fusion rather than by
        // RAW preparation. Include diagnostic options too so a new diagnostic
        // request cannot accidentally reuse a result from the old settings.
        let mut options = std::env::vars_os()
            .filter(|(name, _)| {
                name.to_str().is_some_and(|name| {
                    name.starts_with("RAW_EDITOR_FOCUS_")
                        || name.starts_with("RAW_EDITOR_STATION_")
                        || name == "RAW_EDITOR_STACK_DISABLE_NATIVE_PATCH_REFINEMENT"
                        || name == "RAW_EDITOR_FLAT_ARTWORK_TONE_MAX_ADJUSTMENT"
                })
            })
            .collect::<Vec<_>>();
        options.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        hash.len(options.len());
        for (name, value) in options {
            hash.bytes(name.as_encoded_bytes());
            hash.bytes(value.as_encoded_bytes());
        }
        NativeFusionKey(hash.finish())
    }
}

struct Encoder(Sha256);

impl Encoder {
    fn new(domain: &[u8]) -> Self {
        let mut result = Self(Sha256::new());
        result.bytes(domain);
        result
    }

    fn finish(self) -> [u8; 32] {
        self.0.finalize().into()
    }

    fn u32(&mut self, value: u32) {
        self.0.update(value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.0.update(value.to_le_bytes());
    }

    fn len(&mut self, value: usize) {
        self.u64(value as u64);
    }

    fn flag(&mut self, value: bool) {
        self.0.update([u8::from(value)]);
    }

    fn f64(&mut self, value: f64) {
        self.u64(value.to_bits());
    }

    fn option_f64(&mut self, value: Option<f64>) {
        self.flag(value.is_some());
        if let Some(value) = value {
            self.f64(value);
        }
    }

    fn bytes(&mut self, value: &[u8]) {
        self.len(value.len());
        self.0.update(value);
    }

    fn gray(&mut self, image: &GrayImage) {
        self.u32(image.width());
        self.u32(image.height());
        self.bytes(image.as_raw());
    }

    fn features(&mut self, features: &[Feature]) {
        self.len(features.len());
        for feature in features {
            self.u32(feature.keypoint.x);
            self.u32(feature.keypoint.y);
            self.bytes(&feature.descriptor);
            self.u32(feature.support_scale.to_bits());
        }
    }

    fn matrices(&mut self, values: &HashMap<usize, Matrix3<f64>>) {
        let mut entries = values.iter().collect::<Vec<_>>();
        entries.sort_unstable_by_key(|&(id, _)| *id);
        self.len(entries.len());
        for (&id, matrix) in entries {
            self.len(id);
            for &value in matrix.iter() {
                self.f64(value);
            }
        }
    }

    fn ranges(&mut self, values: &HashMap<usize, (f64, f64)>) {
        let mut entries = values.iter().collect::<Vec<_>>();
        entries.sort_unstable_by_key(|&(id, _)| *id);
        self.len(entries.len());
        for (&id, &(lower, upper)) in entries {
            self.len(id);
            self.f64(lower);
            self.f64(upper);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panorama_utils::shared::{FocusWarpBand, KeyPoint};
    use image::{Luma, Rgb, Rgb32FImage};

    fn source(id: usize) -> ImageInfo {
        ImageInfo {
            id,
            filename: format!("source-{id}.NEF"),
            width: 8,
            height: 6,
            alignment_image: GrayImage::from_pixel(4, 3, Luma([127])),
            full_image: None,
            scale_factor: 0.5,
            focal_length_35mm: Some(50.0),
            source_footprint: None,
            overview_reference: false,
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: Vec::new(),
            vertical_edge_columns: Vec::new(),
        }
    }

    fn geometry() -> HashMap<usize, Matrix3<f64>> {
        HashMap::from([(0, Matrix3::identity()), (1, Matrix3::identity())])
    }

    fn band() -> FocusWarpBand {
        FocusWarpBand {
            homographies: geometry(),
            source_ranges: HashMap::from([(0, (0.0, 0.5)), (1, (0.2, 0.8))]),
            source_x_ranges: HashMap::from([(0, (0.1, 0.9)), (1, (0.0, 1.0))]),
            relax_foreground_seam: false,
            foreground_only: false,
            physical_edge: false,
        }
    }

    #[test]
    fn native_fusion_key_canonicalizes_maps_but_preserves_member_order_and_identity() {
        let (left, right) = (source(0), source(1));
        let first = NativeFusionSourceKey::new(&[&left, &right], [7; 32]);
        let other = NativeFusionSourceKey::new(&[&left, &right], [7; 32]);
        let matrices = geometry();
        let reversed = HashMap::from([(1, Matrix3::identity()), (0, Matrix3::identity())]);
        let warp = FocusLayerWarp {
            bands: vec![band()],
        };
        let mut reordered = warp.clone();
        // The two maps have independently randomized iteration order.
        reordered.bands[0].source_ranges = HashMap::from([(1, (0.2, 0.8)), (0, (0.0, 0.5))]);
        assert_eq!(
            first.cache_key(&matrices, Projection::Planar, Some(&warp), 0, 17),
            other.cache_key(&reversed, Projection::Planar, Some(&reordered), 0, 17)
        );
        assert_ne!(
            first.0,
            NativeFusionSourceKey::new(&[&right, &left], [7; 32]).0
        );
        assert_ne!(
            first.0,
            NativeFusionSourceKey::new(&[&left, &right], [8; 32]).0
        );
    }

    #[test]
    fn native_fusion_key_invalidates_geometry_projection_processing_and_station_changes() {
        let source = source(0);
        let inputs = NativeFusionSourceKey::new(&[&source], [0; 32]);
        let matrices = geometry();
        let baseline = inputs.cache_key(&matrices, Projection::Planar, None, 0, 0);
        let mut repaired = matrices.clone();
        repaired.get_mut(&0).unwrap()[(0, 2)] = 0.125;
        assert_ne!(
            baseline,
            inputs.cache_key(&repaired, Projection::Planar, None, 0, 0)
        );
        repaired.remove(&0);
        assert_ne!(
            baseline,
            inputs.cache_key(&repaired, Projection::Planar, None, 0, 0)
        );
        assert_ne!(
            baseline,
            inputs.cache_key(&matrices, Projection::Spherical, None, 0, 0)
        );
        assert_ne!(
            baseline,
            inputs.cache_key(&matrices, Projection::Planar, None, 1, 0)
        );
        assert_ne!(
            baseline,
            inputs.cache_key(&matrices, Projection::Planar, None, 0, 1)
        );
    }

    #[test]
    fn native_fusion_key_invalidates_band_ranges_flags_and_order() {
        let source = source(0);
        let inputs = NativeFusionSourceKey::new(&[&source], [0; 32]);
        let matrices = geometry();
        let warp = FocusLayerWarp {
            bands: vec![band()],
        };
        let baseline = inputs.cache_key(&matrices, Projection::Planar, Some(&warp), 0, 0);
        let changes: Vec<Box<dyn Fn(&mut FocusWarpBand)>> = vec![
            Box::new(|b| b.homographies.get_mut(&0).unwrap()[(1, 2)] = 0.25),
            Box::new(|b| {
                b.source_ranges.insert(0, (0.0, 0.6));
            }),
            Box::new(|b| {
                b.source_x_ranges.insert(0, (0.0, 0.8));
            }),
            Box::new(|b| b.relax_foreground_seam = true),
            Box::new(|b| b.foreground_only = true),
            Box::new(|b| b.physical_edge = true),
        ];
        for change in changes {
            let mut changed = warp.clone();
            change(&mut changed.bands[0]);
            assert_ne!(
                baseline,
                inputs.cache_key(&matrices, Projection::Planar, Some(&changed), 0, 0)
            );
        }
        let mut second = band();
        second.foreground_only = true;
        let ordered = FocusLayerWarp {
            bands: vec![band(), second.clone()],
        };
        let reversed = FocusLayerWarp {
            bands: vec![second, band()],
        };
        assert_ne!(
            inputs.cache_key(&matrices, Projection::Planar, Some(&ordered), 0, 0),
            inputs.cache_key(&matrices, Projection::Planar, Some(&reversed), 0, 0)
        );
    }

    #[test]
    fn native_fusion_source_snapshot_tracks_pixels_masks_features_and_support_metadata() {
        let original = source(0);
        let baseline = NativeFusionSourceKey::new(&[&original], [0; 32]).0;
        let changes: Vec<Box<dyn Fn(&mut ImageInfo)>> = vec![
            Box::new(|s| s.alignment_image.put_pixel(2, 1, Luma([128]))),
            Box::new(|s| s.foreground_mask = Some(GrayImage::from_pixel(4, 3, Luma([255])))),
            Box::new(|s| s.full_image = Some(Rgb32FImage::from_pixel(8, 6, Rgb([0.1, 0.2, 0.3])))),
            Box::new(|s| {
                s.features.push(Feature {
                    keypoint: KeyPoint { x: 1, y: 1 },
                    descriptor: [3; 32],
                    support_scale: 1.0,
                })
            }),
            Box::new(|s| {
                s.top_features.push(Feature {
                    keypoint: KeyPoint { x: 1, y: 1 },
                    descriptor: [3; 32],
                    support_scale: 1.0,
                })
            }),
            Box::new(|s| s.foreground_range = Some((0.1, 0.9))),
            Box::new(|s| s.horizontal_edge_rows.push(1.0)),
            Box::new(|s| s.vertical_edge_columns.push(2.0)),
            Box::new(|s| s.overview_reference = true),
            Box::new(|s| s.width += 1),
            Box::new(|s| s.scale_factor = 0.25),
            Box::new(|s| s.focal_length_35mm = Some(55.0)),
            Box::new(|s| s.filename = "replacement.NEF".into()),
        ];
        for change in changes {
            let mut changed = source(0);
            change(&mut changed);
            assert_ne!(baseline, NativeFusionSourceKey::new(&[&changed], [0; 32]).0);
        }
        let mut first = source(0);
        first.full_image = Some(Rgb32FImage::from_pixel(8, 6, Rgb([0.1, 0.2, 0.3])));
        let before = NativeFusionSourceKey::new(&[&first], [0; 32]).0;
        first
            .full_image
            .as_mut()
            .unwrap()
            .put_pixel(3, 2, Rgb([0.2, 0.2, 0.3]));
        assert_ne!(before, NativeFusionSourceKey::new(&[&first], [0; 32]).0);
    }
}
