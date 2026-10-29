//! Image-stack output formats and the fidelity notices they imply.
//!
//! Parses the requested TIFF/PNG/JPEG format, maps it to its encoded colour type
//! and reports the bit-depth and transparency downgrades it implies.

use image::ColorType;
use image::DynamicImage;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ImageStackOutputFormat {
    Tiff,
    Png,
    Jpeg,
}

impl ImageStackOutputFormat {
    pub(super) fn from_wire(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "tif" | "tiff" => Ok(Self::Tiff),
            "png" => Ok(Self::Png),
            "jpg" | "jpeg" => Ok(Self::Jpeg),
            _ => Err(format!(
                "Unsupported image-stack output format '{value}'. Choose TIFF, PNG, or JPEG."
            )),
        }
    }

    pub(super) fn from_extension(extension: &str) -> Option<Self> {
        match extension.to_ascii_lowercase().as_str() {
            "tif" | "tiff" => Some(Self::Tiff),
            "png" => Some(Self::Png),
            "jpg" | "jpeg" => Some(Self::Jpeg),
            _ => None,
        }
    }

    pub(super) fn canonical_extension(self) -> &'static str {
        match self {
            Self::Tiff => "tiff",
            Self::Png => "png",
            Self::Jpeg => "jpg",
        }
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Tiff => "TIFF",
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
        }
    }

    /// TIFF and PNG carry the Coverage_Mask as alpha; JPEG cannot (需求 10.6).
    pub(super) fn keeps_alpha(self) -> bool {
        matches!(self, Self::Tiff | Self::Png)
    }

    pub(super) fn encoded_color_type(self, bit_depth: u8, alpha: bool) -> ColorType {
        match (self, bit_depth, alpha && self.keeps_alpha()) {
            (Self::Tiff | Self::Png, 16, true) => ColorType::Rgba16,
            (Self::Tiff | Self::Png, 16, false) => ColorType::Rgb16,
            (Self::Tiff | Self::Png, _, true) => ColorType::Rgba8,
            _ => ColorType::Rgb8,
        }
    }
}

/// What an export keeps of the canonical result (需求 10.5 / 10.6 / 10.10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct OutputFidelity {
    pub(super) bit_depth: u8,
    /// The result has uncovered, transparent pixels.
    pub(super) has_transparency: bool,
    pub(super) alpha_preserved: bool,
}

/// A localized export warning.  The backend owns only the stable identifier
/// and interpolation values; user-facing wording belongs to the frontend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OutputDowngradeNotice {
    pub id: String,
    pub params: OutputDowngradeParams,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OutputDowngradeParams {
    pub format: String,
    #[serde(rename = "bitDepth", skip_serializing_if = "Option::is_none")]
    pub bit_depth: Option<u8>,
}

impl OutputFidelity {
    pub(super) fn of(
        image: &DynamicImage,
        output_format: ImageStackOutputFormat,
        requested_bit_depth: u8,
    ) -> Self {
        let has_transparency = match image {
            DynamicImage::ImageRgba16(pixels) => pixels.pixels().any(|pixel| pixel[3] < u16::MAX),
            DynamicImage::ImageRgba8(pixels) => pixels.pixels().any(|pixel| pixel[3] < u8::MAX),
            _ => image.color().has_alpha() && image.to_rgba16().pixels().any(|p| p[3] < u16::MAX),
        };
        Self {
            bit_depth: crate::export_processing::effective_export_bit_depth(
                output_format.canonical_extension(),
                requested_bit_depth,
            ),
            has_transparency,
            alpha_preserved: output_format.keeps_alpha() && image.color().has_alpha(),
        }
    }

    /// The downgrade notices to show before writing, each led by its stable
    /// identifier (需求 10.10): fewer than 16 bits per channel, or
    /// transparent pixels that the format cannot keep.
    pub(super) fn downgrades(
        &self,
        output_format: ImageStackOutputFormat,
    ) -> Vec<OutputDowngradeNotice> {
        let mut notices = Vec::new();
        if self.bit_depth < 16 {
            notices.push(OutputDowngradeNotice {
                id: crate::panorama_utils::stack_pipeline::degradation::OUTPUT_BIT_DEPTH_DOWNGRADED
                    .to_string(),
                params: OutputDowngradeParams {
                    format: output_format.label().to_string(),
                    bit_depth: Some(self.bit_depth),
                },
            });
        }
        if self.has_transparency && !self.alpha_preserved {
            notices.push(OutputDowngradeNotice {
                id: crate::panorama_utils::stack_pipeline::degradation::OUTPUT_ALPHA_UNSUPPORTED
                    .to_string(),
                params: OutputDowngradeParams {
                    format: output_format.label().to_string(),
                    bit_depth: None,
                },
            });
        }
        notices
    }
}
