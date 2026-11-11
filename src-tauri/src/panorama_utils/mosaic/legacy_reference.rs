//! Radiometric provenance for direct legacy source references. These fields
//! are copied from rendering, never fitted to the delivered QualityGate ROI.
use super::{Field, adjusted};
use image::{Rgb, Rgb32FImage};
use nalgebra::Point2;
use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

#[derive(Clone, Debug)]
pub(crate) struct MosaicGainReference {
    pub gains: Arc<Rgb32FImage>,
    pub crop: (u32, u32, u32, u32),
    pub scale: f64,
    pub max_log: f32,
}
#[derive(Clone, Debug)]
pub(crate) struct MosaicReferenceTone {
    pub field: Field<3>,
    pub left: u32,
    pub top: u32,
    pub scale: f64,
    pub offset: (f64, f64),
    pub post_gain: Option<MosaicGainReference>,
    /// The focus-scan exposure plane the frame was loaded with (canvas
    /// coordinates), applied before the layer tone as in rendering.
    pub frame_exposure: Option<super::frame_exposure::FrameExposure>,
}
impl MosaicReferenceTone {
    pub(crate) fn apply(&self, world: Point2<f64>, pixel: Rgb<f32>) -> Rgb<f32> {
        let x = world.x + self.offset.0;
        let y = world.y + self.offset.1;
        let mut pixel = pixel;
        if let Some(exposure) = &self.frame_exposure {
            let gain = exposure.log_gain_at((x, y));
            for (value, gain) in pixel.0.iter_mut().zip(gain) {
                *value *= gain.exp() as f32;
            }
        }
        let mut pixel = adjusted(
            pixel,
            self.field.at(
                (x - f64::from(self.left)) / self.scale,
                (y - f64::from(self.top)) / self.scale,
            ),
        );
        if let Some(gain) = &self.post_gain {
            let log = super::super::stitching::get_high_quality_interpolated_pixel(
                &gain.gains,
                (x - f64::from(gain.crop.0)) / gain.scale,
                (y - f64::from(gain.crop.1)) / gain.scale,
            )[0]
            .clamp(-gain.max_log, gain.max_log);
            for value in &mut pixel.0 {
                *value = (*value * log.exp()).clamp(0.0, 1.0);
            }
        }
        pixel
    }
    pub(crate) fn cache_fingerprint(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.left.hash(&mut h);
        self.top.hash(&mut h);
        self.scale.to_bits().hash(&mut h);
        self.offset.0.to_bits().hash(&mut h);
        self.offset.1.to_bits().hash(&mut h);
        self.field.width.hash(&mut h);
        self.field.height.hash(&mut h);
        self.field.step.to_bits().hash(&mut h);
        for row in &self.field.values {
            for value in row {
                value.to_bits().hash(&mut h);
            }
        }
        if let Some(exposure) = &self.frame_exposure {
            exposure.hash_into(&mut h);
        }
        if let Some(gain) = &self.post_gain {
            // Immutable and run-local; Arc identity invalidates the ROI cache
            // without hashing the same 2048px gain image for all 45 sources.
            (Arc::as_ptr(&gain.gains) as usize).hash(&mut h);
            gain.crop.hash(&mut h);
            gain.scale.to_bits().hash(&mut h);
            gain.max_log.to_bits().hash(&mut h);
        }
        h.finish()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_tone_and_final_gain_replay_in_render_order() {
        let mut field = Field::new(64, 64, 112.0);
        field.values.fill([0.8_f64.ln(), 1.1_f64.ln(), 0.0]);
        let tone = MosaicReferenceTone {
            field,
            left: 40,
            top: 10,
            scale: 2.0,
            offset: (37.0, -19.0),
            post_gain: Some(MosaicGainReference {
                gains: Arc::new(Rgb32FImage::from_pixel(4, 4, Rgb([1.04_f32.ln(); 3]))),
                crop: (0, 0, 100, 100),
                scale: 32.0,
                max_log: 1.08_f32.ln(),
            }),
            frame_exposure: None,
        };
        let actual = tone.apply(Point2::new(50.0, 50.0), Rgb([0.5, 0.8, 0.9]));
        for (value, expected) in
            actual
                .0
                .into_iter()
                .zip([0.5 * 0.8 * 1.04, 0.8 * 1.1 * 1.04, 0.9 * 1.04])
        {
            assert!((value - expected).abs() < 1e-6);
        }
    }
    #[test]
    fn frame_exposure_replays_before_the_layer_tone() {
        // Rendering scales the loaded frame by its exposure plane at the
        // canvas position, then applies the layer tone; replay must match.
        let exposure = super::super::frame_exposure::FrameExposure::with_planes(
            (100.0, 50.0),
            (50.0, 25.0),
            [[0.1, 0.05, 0.0], [0.0, 0.0, 0.0], [-0.1, 0.0, 0.02]],
        );
        let mut field = Field::new(64, 64, 112.0);
        field.values.fill([0.0, 0.9_f64.ln(), 0.0]);
        let tone = MosaicReferenceTone {
            field,
            left: 0,
            top: 0,
            scale: 1.0,
            offset: (20.0, 10.0),
            post_gain: None,
            frame_exposure: Some(exposure),
        };
        let world = Point2::new(105.0, 52.5);
        let gain = exposure.log_gain_at((world.x + 20.0, world.y + 10.0));
        let source = Rgb([0.4_f32, 0.5, 0.6]);
        let actual = tone.apply(world, source);
        let expected = [
            0.4 * gain[0].exp(),
            0.5 * gain[1].exp() * 0.9,
            0.6 * gain[2].exp(),
        ];
        for (value, expected) in actual.0.into_iter().zip(expected) {
            assert!(
                (f64::from(value) - expected).abs() < 1e-6,
                "{value} != {expected}"
            );
        }
        let mut without = tone.clone();
        without.frame_exposure = None;
        assert_ne!(tone.cache_fingerprint(), without.cache_fingerprint());
    }
    #[test]
    fn reference_cache_identity_includes_the_replayed_source_tone() {
        let mut tone = MosaicReferenceTone {
            field: Field::new(64, 64, 112.0),
            left: 0,
            top: 0,
            scale: 1.0,
            offset: (0.0, 0.0),
            post_gain: None,
            frame_exposure: None,
        };
        let old = tone.cache_fingerprint();
        tone.field.values[0][0] = 0.03;
        assert_ne!(old, tone.cache_fingerprint());
    }
}
