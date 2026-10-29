//! Strided sample summaries of RAW mosaics and developed images.
//!
//! At most `MAX_SAMPLED_PIXELS` values per buffer are hashed and summarised.

use image::DynamicImage;
use image::GenericImageView;
use rawler::rawimage::RawImageData;
use sha2::Digest;
use sha2::Sha256;

use super::{Dimensions, ImageSummary, RawSampleSummary};

pub(super) const MAX_SAMPLED_PIXELS: usize = 262_144;

pub(super) fn summarize_raw_samples(data: &RawImageData) -> RawSampleSummary {
    match data {
        RawImageData::Integer(values) => {
            let stride = sampling_stride(values.len());
            let mut hasher = Sha256::new();
            let mut count = 0usize;
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            let mut sum = 0.0f64;
            for value in values.iter().step_by(stride) {
                hasher.update(value.to_le_bytes());
                let value = *value as f32;
                min = min.min(value);
                max = max.max(value);
                sum += value as f64;
                count += 1;
            }
            RawSampleSummary {
                sha256: hex::encode(hasher.finalize()),
                values: count,
                min,
                max,
                mean: sum / count as f64,
            }
        }
        RawImageData::Float(values) => {
            let stride = sampling_stride(values.len());
            let mut hasher = Sha256::new();
            let mut count = 0usize;
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            let mut sum = 0.0f64;
            for value in values.iter().step_by(stride) {
                assert!(value.is_finite(), "RAW mosaic contains a non-finite value");
                hasher.update(value.to_bits().to_le_bytes());
                min = min.min(*value);
                max = max.max(*value);
                sum += *value as f64;
                count += 1;
            }
            RawSampleSummary {
                sha256: hex::encode(hasher.finalize()),
                values: count,
                min,
                max,
                mean: sum / count as f64,
            }
        }
    }
}

pub(super) fn summarize_image(image: &DynamicImage, label: &str) -> ImageSummary {
    let dimensions = Dimensions::from_u32(image.dimensions());
    let total_pixels = dimensions.width as usize * dimensions.height as usize;
    let stride = sampling_stride(total_pixels);
    let mut hasher = Sha256::new();
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut sums = [0.0f64; 3];
    let mut sampled_pixels = 0usize;

    let mut consume = |rgb: [f32; 3]| {
        for channel in 0..3 {
            assert!(
                rgb[channel].is_finite(),
                "{label} contains a non-finite value"
            );
            hasher.update(rgb[channel].to_bits().to_le_bytes());
            min[channel] = min[channel].min(rgb[channel]);
            max[channel] = max[channel].max(rgb[channel]);
            sums[channel] += rgb[channel] as f64;
        }
        sampled_pixels += 1;
    };

    match image {
        DynamicImage::ImageRgb32F(buffer) => {
            let values = buffer.as_raw();
            for pixel in (0..total_pixels).step_by(stride) {
                let offset = pixel * 3;
                consume([values[offset], values[offset + 1], values[offset + 2]]);
            }
        }
        DynamicImage::ImageRgba32F(buffer) => {
            let values = buffer.as_raw();
            for pixel in (0..total_pixels).step_by(stride) {
                let offset = pixel * 4;
                consume([values[offset], values[offset + 1], values[offset + 2]]);
            }
        }
        _ => {
            let converted = image.to_rgb32f();
            let values = converted.as_raw();
            for pixel in (0..total_pixels).step_by(stride) {
                let offset = pixel * 3;
                consume([values[offset], values[offset + 1], values[offset + 2]]);
            }
        }
    }

    ImageSummary {
        dimensions,
        sampled_pixels,
        sampled_pixel_sha256: hex::encode(hasher.finalize()),
        min,
        max,
        channel_means: sums.map(|sum| sum / sampled_pixels as f64),
    }
}

pub(super) fn sampling_stride(values: usize) -> usize {
    values.div_ceil(MAX_SAMPLED_PIXELS).max(1)
}
