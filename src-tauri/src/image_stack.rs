use crate::app_state::AppState;
use crate::export_processing::ExportSettings;
use crate::file_management::parse_virtual_path;
use crate::panorama_stitching::{AlignmentMode, BlendMode, stitch_images_with_options};
use crate::panorama_utils::stack_pipeline::degradation::{
    DegradationLedger, DegradationManager, OutcomeDecision, OutputPublication,
    StandardFinalOutputFileSystem,
};
use image::codecs::jpeg::{JpegDecoder, JpegEncoder};
use image::codecs::png::{PngDecoder, PngEncoder};
use image::codecs::tiff::{TiffDecoder, TiffEncoder};
use image::imageops::FilterType;
use image::{ColorType, DynamicImage, GenericImageView, ImageDecoder, ImageEncoder, RgbImage};
use serde::Serialize;
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use tauri::{AppHandle, Emitter, Manager};
use tempfile::NamedTempFile;
use uuid::Uuid;

// Keep the UI proxy comfortably inside desktop WebView image-decoding budgets.
// The full-resolution result remains in AppState and is used for TIFF export.
const PREVIEW_MAX_LONG_SIDE: u32 = 4_096;
const PREVIEW_MAX_PIXELS: u64 = 12_000_000;
const PREVIEW_JPEG_QUALITY: u8 = 92;
const DETAIL_PREVIEW_MAX_LONG_SIDE: u32 = 8_192;
const DETAIL_PREVIEW_MAX_PIXELS: u64 = 32_000_000;
const DETAIL_PREVIEW_JPEG_QUALITY: u8 = 98;
#[cfg(test)]
const IMAGE_STACK_JPEG_QUALITY: u8 = 95;
// Large artwork scans commonly contain several focus distances at every camera
// position. The stitching backend keeps preparation and candidate matching
// bounded, so reject only genuinely exceptional selections rather than forcing
// a 382-frame scan to be split into independently warped mosaics.
const IMAGE_STACK_MAX_SOURCES: usize = 500;
const IMAGE_STACK_PIPELINE_VERSION: &str = "image-stack-2026.09.16.1";
// Identifies the layered Stack_Pipeline itself rather than the frontend/backend
// handshake.  It is one of the three ingredients of the Virtual_Tile cache key
// (source paths, source SHA-256 set, pipeline version), so *any* change to a
// grouping, registration, fusion, geometry, tone or composition stage must bump
// it or a stale cache entry will be replayed.
pub(crate) const STACK_PIPELINE_VERSION: &str = "stack-2026.09.22.2";

fn validate_image_stack_source_count(count: usize) -> Result<(), String> {
    if count < 2 {
        return Err("Please select at least two images.".to_string());
    }
    if count > IMAGE_STACK_MAX_SOURCES {
        return Err(format!(
            "Image stack is limited to {IMAGE_STACK_MAX_SOURCES} source images."
        ));
    }
    Ok(())
}

fn validate_image_stack_pipeline_version(value: &str) -> Result<(), String> {
    if value == IMAGE_STACK_PIPELINE_VERSION {
        return Ok(());
    }

    Err(format!(
        "The image-stack backend is out of date (expected {IMAGE_STACK_PIPELINE_VERSION}, received {}). Fully quit and restart RAW Editor before stacking again.",
        if value.trim().is_empty() {
            "no version"
        } else {
            value
        }
    ))
}

fn resolve_blend_mode(value: &str) -> BlendMode {
    match value {
        "focus" => BlendMode::FocusStack,
        _ => BlendMode::Panorama,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImageStackOutputFormat {
    Tiff,
    Png,
    Jpeg,
}

impl ImageStackOutputFormat {
    fn from_wire(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "tif" | "tiff" => Ok(Self::Tiff),
            "png" => Ok(Self::Png),
            "jpg" | "jpeg" => Ok(Self::Jpeg),
            _ => Err(format!(
                "Unsupported image-stack output format '{value}'. Choose TIFF, PNG, or JPEG."
            )),
        }
    }

    fn from_extension(extension: &str) -> Option<Self> {
        match extension.to_ascii_lowercase().as_str() {
            "tif" | "tiff" => Some(Self::Tiff),
            "png" => Some(Self::Png),
            "jpg" | "jpeg" => Some(Self::Jpeg),
            _ => None,
        }
    }

    fn canonical_extension(self) -> &'static str {
        match self {
            Self::Tiff => "tiff",
            Self::Png => "png",
            Self::Jpeg => "jpg",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Tiff => "TIFF",
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
        }
    }

    /// TIFF and PNG carry the Coverage_Mask as alpha; JPEG cannot (需求 10.6).
    fn keeps_alpha(self) -> bool {
        matches!(self, Self::Tiff | Self::Png)
    }

    fn encoded_color_type(self, bit_depth: u8, alpha: bool) -> ColorType {
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
struct OutputFidelity {
    bit_depth: u8,
    /// The result has uncovered, transparent pixels.
    has_transparency: bool,
    alpha_preserved: bool,
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
    fn of(
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
    fn downgrades(&self, output_format: ImageStackOutputFormat) -> Vec<OutputDowngradeNotice> {
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

fn bounded_preview_dimensions(
    width: u32,
    height: u32,
    max_long_side: u32,
    max_pixels: u64,
) -> (u32, u32) {
    if width == 0 || height == 0 {
        return (0, 0);
    }
    let long_side_scale = max_long_side as f64 / width.max(height) as f64;
    let pixel_scale = (max_pixels as f64 / (width as u64 * height as u64) as f64).sqrt();
    let scale = 1.0_f64.min(long_side_scale).min(pixel_scale);
    (
        (width as f64 * scale).round().max(1.0) as u32,
        (height as f64 * scale).round().max(1.0) as u32,
    )
}

fn preview_dimensions(width: u32, height: u32) -> (u32, u32) {
    bounded_preview_dimensions(width, height, PREVIEW_MAX_LONG_SIDE, PREVIEW_MAX_PIXELS)
}

fn detail_preview_dimensions(width: u32, height: u32) -> (u32, u32) {
    bounded_preview_dimensions(
        width,
        height,
        DETAIL_PREVIEW_MAX_LONG_SIDE,
        DETAIL_PREVIEW_MAX_PIXELS,
    )
}

/// Image stacking works on display-referred samples after RAW development. Freeze
/// those samples once at 16-bit precision so preview and export cannot diverge by
/// independently interpreting a floating-point TIFF as linear RGB.
/// The opaque canonical form: production canonicalises through
/// [`canonicalize_image_stack_result_with_coverage`]; tests and the gate
/// harness use this RGB-only form.
#[cfg(test)]
pub(crate) fn canonicalize_image_stack_result(image: DynamicImage) -> DynamicImage {
    DynamicImage::ImageRgb16(image.to_rgb16())
}

/// The canonical result with the Coverage_Mask as alpha (需求 10.3 / 10.6):
/// uncovered pixels fully transparent, covered pixels fully opaque, and the
/// colour channels exactly those of [`canonicalize_image_stack_result`]. A
/// missing or mismatched mask keeps the opaque RGB canonical form.
pub(crate) fn canonicalize_image_stack_result_with_coverage(
    image: DynamicImage,
    coverage: Option<&image::GrayImage>,
) -> DynamicImage {
    let rgb = image.to_rgb16();
    let Some(coverage) = coverage.filter(|mask| mask.dimensions() == rgb.dimensions()) else {
        return DynamicImage::ImageRgb16(rgb);
    };
    let rgba = image::ImageBuffer::from_fn(rgb.width(), rgb.height(), |x, y| {
        let [red, green, blue] = rgb.get_pixel(x, y).0;
        let alpha = if coverage.get_pixel(x, y)[0] == 0 {
            0
        } else {
            u16::MAX
        };
        image::Rgba([red, green, blue, alpha])
    });
    DynamicImage::ImageRgba16(rgba)
}

/// The pixels a lossless format stores: RGBA when the result carries its
/// Coverage_Mask as alpha, RGB otherwise; colour channels are converted the
/// same way either way, so writing alpha never changes them (需求 10.6).
fn lossless_pixels(image: &DynamicImage, bit_depth: u8) -> DynamicImage {
    match (bit_depth == 16, image.color().has_alpha()) {
        (true, true) => DynamicImage::ImageRgba16(image.to_rgba16()),
        (true, false) => DynamicImage::ImageRgb16(image.to_rgb16()),
        (false, true) => DynamicImage::ImageRgba8(image.to_rgba8()),
        (false, false) => DynamicImage::ImageRgb8(image.to_rgb8()),
    }
}

fn encode_srgb_image_stack<W: Write + Seek>(
    image: &DynamicImage,
    writer: W,
    output_format: ImageStackOutputFormat,
    jpeg_quality: u8,
    bit_depth: u8,
    embed_color_profile: bool,
    export_exif: Option<&[u8]>,
) -> Result<(), String> {
    let bit_depth = crate::export_processing::effective_export_bit_depth(
        output_format.canonical_extension(),
        bit_depth,
    );
    let profile = crate::color_management::srgb_v4_profile().to_vec();
    match output_format {
        ImageStackOutputFormat::Tiff => {
            let mut encoder = TiffEncoder::new(writer);
            if embed_color_profile {
                encoder.set_icc_profile(profile).map_err(|error| {
                    format!("Failed to attach the image-stack TIFF color profile: {error}")
                })?;
            }
            let image_to_encode = lossless_pixels(image, bit_depth);
            image_to_encode
                .write_with_encoder(encoder)
                .map_err(|error| format!("Failed to encode image-stack TIFF: {error}"))
        }
        ImageStackOutputFormat::Png => {
            let mut encoder = PngEncoder::new(writer);
            if embed_color_profile {
                encoder.set_icc_profile(profile).map_err(|error| {
                    format!("Failed to attach the image-stack PNG color profile: {error}")
                })?;
            }
            if let Some(exif) = export_exif {
                encoder.set_exif_metadata(exif.to_vec()).map_err(|error| {
                    format!("Failed to attach image-stack PNG metadata: {error}")
                })?;
            }
            let image_to_encode = lossless_pixels(image, bit_depth);
            image_to_encode
                .write_with_encoder(encoder)
                .map_err(|error| format!("Failed to encode image-stack PNG: {error}"))
        }
        ImageStackOutputFormat::Jpeg => {
            let mut encoder = JpegEncoder::new_with_quality(writer, jpeg_quality.clamp(1, 100));
            if embed_color_profile {
                encoder.set_icc_profile(profile).map_err(|error| {
                    format!("Failed to attach the image-stack JPEG color profile: {error}")
                })?;
            }
            if let Some(exif) = export_exif {
                encoder.set_exif_metadata(exif.to_vec()).map_err(|error| {
                    format!("Failed to attach image-stack JPEG metadata: {error}")
                })?;
            }
            encoder
                .encode_image(&image.to_rgb8())
                .map_err(|error| format!("Failed to encode image-stack JPEG: {error}"))
        }
    }
}

#[cfg(test)]
fn encode_srgb_tiff<W: Write + Seek>(image: &DynamicImage, writer: W) -> Result<(), String> {
    encode_srgb_image_stack(
        image,
        writer,
        ImageStackOutputFormat::Tiff,
        95,
        16,
        true,
        None,
    )
}

#[cfg(not(target_os = "android"))]
fn encode_srgb_jpeg_streaming(
    image: &DynamicImage,
    output: &mut File,
    jpeg_quality: u8,
    embed_color_profile: bool,
    export_exif: Option<&[u8]>,
) -> Result<(), String> {
    // JPEG has no alpha (需求 10.10 notices this before writing): drop the
    // Coverage_Mask channel, keeping the colour samples unchanged.
    let without_alpha;
    let rgb16 = match image.as_rgb16() {
        Some(rgb16) => rgb16,
        None if image.color().has_alpha() => {
            without_alpha = image.to_rgb16();
            &without_alpha
        }
        None => return Err("The canonical image-stack result is not RGB16.".to_string()),
    };
    let (width, height) = rgb16.dimensions();
    let source_row_samples = (width as usize)
        .checked_mul(3)
        .ok_or_else(|| "The image-stack JPEG row is too wide to encode.".to_string())?;
    let rgba_row_bytes = (width as usize)
        .checked_mul(4)
        .ok_or_else(|| "The image-stack JPEG row is too wide to encode.".to_string())?;
    let mut rgba_row = vec![0_u8; rgba_row_bytes];

    crate::export_processing::encode_streaming_jpeg(
        output,
        width,
        height,
        jpeg_quality,
        embed_color_profile,
        export_exif,
        |sink| {
            for source_row in rgb16.as_raw().chunks_exact(source_row_samples) {
                for (source, target) in source_row.chunks_exact(3).zip(rgba_row.chunks_exact_mut(4))
                {
                    target[0] = ((u32::from(source[0]) + 128) / 257) as u8;
                    target[1] = ((u32::from(source[1]) + 128) / 257) as u8;
                    target[2] = ((u32::from(source[2]) + 128) / 257) as u8;
                    target[3] = 255;
                }
                sink(&rgba_row)?;
            }
            Ok(())
        },
    )
}

fn encode_image_stack_file(
    image: &DynamicImage,
    output: &mut File,
    output_format: ImageStackOutputFormat,
    export_settings: &ExportSettings,
    source_path: &str,
) -> Result<(), String> {
    let bit_depth = crate::export_processing::effective_export_bit_depth(
        output_format.canonical_extension(),
        export_settings.bit_depth,
    );
    let export_exif = if output_format == ImageStackOutputFormat::Tiff {
        None
    } else {
        crate::exif_processing::export_metadata_tiff_payload(
            source_path,
            output_format.canonical_extension(),
            export_settings.keep_metadata,
            export_settings.strip_gps,
            export_settings.metadata_overrides.as_ref(),
        )?
    };

    #[cfg(not(target_os = "android"))]
    if output_format == ImageStackOutputFormat::Jpeg {
        encode_srgb_jpeg_streaming(
            image,
            output,
            export_settings.jpeg_quality,
            export_settings.embed_color_profile,
            export_exif.as_deref(),
        )?;
        return output
            .flush()
            .map_err(|error| format!("Failed to finish writing the image-stack JPEG: {error}"));
    }

    #[cfg(not(target_os = "android"))]
    if output_format == ImageStackOutputFormat::Tiff {
        let metadata = crate::exif_processing::export_metadata_for_streaming_tiff(
            source_path,
            export_settings.keep_metadata,
            export_settings.strip_gps,
            export_settings.metadata_overrides.as_ref(),
        )?;
        // The Coverage_Mask travels as alpha; the colour samples are the same
        // either way (需求 10.6).
        if image.color().has_alpha() {
            if bit_depth == 16 {
                let rgba16 = image.to_rgba16();
                return crate::export_processing::encode_rgba16_tiff_with_metadata(
                    output,
                    rgba16.width(),
                    rgba16.height(),
                    rgba16.as_raw(),
                    export_settings.embed_color_profile,
                    metadata.as_ref(),
                );
            }
            let rgba8 = image.to_rgba8();
            return crate::export_processing::encode_rgba8_tiff_with_metadata(
                output,
                rgba8.width(),
                rgba8.height(),
                rgba8.as_raw(),
                export_settings.embed_color_profile,
                metadata.as_ref(),
            );
        }
        let rgb16 = image
            .as_rgb16()
            .ok_or_else(|| "The canonical image-stack result is not RGB16.".to_string())?;
        if bit_depth == 16 {
            return crate::export_processing::encode_rgb16_tiff_with_metadata(
                output,
                rgb16.width(),
                rgb16.height(),
                rgb16.as_raw(),
                export_settings.embed_color_profile,
                metadata.as_ref(),
            );
        }
        let rgb8 = image.to_rgb8();
        return crate::export_processing::encode_rgb8_tiff_with_metadata(
            output,
            rgb8.width(),
            rgb8.height(),
            rgb8.as_raw(),
            export_settings.embed_color_profile,
            metadata.as_ref(),
        );
    }

    let mut writer = BufWriter::new(output);
    encode_srgb_image_stack(
        image,
        &mut writer,
        output_format,
        export_settings.jpeg_quality,
        bit_depth,
        export_settings.embed_color_profile,
        export_exif.as_deref(),
    )?;
    writer.flush().map_err(|error| {
        format!(
            "Failed to finish writing the image-stack {}: {error}",
            output_format.label()
        )
    })
}

fn validate_image_stack_decoder<D: ImageDecoder>(
    mut decoder: D,
    dimensions: (u32, u32),
    output_format: ImageStackOutputFormat,
    bit_depth: u8,
    alpha: bool,
    expect_color_profile: bool,
) -> Result<(), String> {
    if decoder.dimensions() != dimensions {
        return Err(format!(
            "Saved image-stack dimensions do not match the result (expected {}×{}, found {}×{}).",
            dimensions.0,
            dimensions.1,
            decoder.dimensions().0,
            decoder.dimensions().1
        ));
    }
    if decoder.color_type() != output_format.encoded_color_type(bit_depth, alpha) {
        return Err(format!(
            "Saved image-stack {} has an unexpected pixel format ({:?}).",
            output_format.label(),
            decoder.color_type()
        ));
    }
    let saved_profile = decoder.icc_profile().map_err(|error| {
        format!(
            "Failed to validate the image-stack {} color profile: {error}",
            output_format.label()
        )
    })?;
    if expect_color_profile {
        if saved_profile.as_deref() != Some(crate::color_management::srgb_v4_profile()) {
            return Err(format!(
                "The saved image-stack {} is missing its sRGB color profile.",
                output_format.label()
            ));
        }
    } else if saved_profile.is_some() {
        return Err(format!(
            "The saved image-stack {} contains a color profile even though embedding was disabled.",
            output_format.label()
        ));
    }
    Ok(())
}

fn validate_image_stack_output(
    output_path: &Path,
    dimensions: (u32, u32),
    output_format: ImageStackOutputFormat,
    bit_depth: u8,
    alpha: bool,
    expect_color_profile: bool,
) -> Result<(), String> {
    let open = || {
        File::open(output_path)
            .map(BufReader::new)
            .map_err(|error| {
                format!(
                    "Failed to reopen the saved image-stack {}: {error}",
                    output_format.label()
                )
            })
    };
    match output_format {
        ImageStackOutputFormat::Tiff => validate_image_stack_decoder(
            TiffDecoder::new(open()?).map_err(|error| {
                format!("Failed to validate the saved image-stack TIFF: {error}")
            })?,
            dimensions,
            output_format,
            bit_depth,
            alpha,
            expect_color_profile,
        ),
        ImageStackOutputFormat::Png => validate_image_stack_decoder(
            PngDecoder::new(open()?).map_err(|error| {
                format!("Failed to validate the saved image-stack PNG: {error}")
            })?,
            dimensions,
            output_format,
            bit_depth,
            alpha,
            expect_color_profile,
        ),
        ImageStackOutputFormat::Jpeg => validate_image_stack_decoder(
            JpegDecoder::new(open()?).map_err(|error| {
                format!("Failed to validate the saved image-stack JPEG: {error}")
            })?,
            dimensions,
            output_format,
            bit_depth,
            alpha,
            expect_color_profile,
        ),
    }
}

#[cfg(test)]
fn default_image_stack_export_settings() -> ExportSettings {
    ExportSettings {
        bit_depth: 16,
        jpeg_quality: IMAGE_STACK_JPEG_QUALITY,
        resize: None,
        keep_metadata: false,
        metadata_overrides: None,
        preserve_timestamps: false,
        strip_gps: true,
        embed_color_profile: true,
        filename_template: None,
        watermark: None,
        export_masks: false,
        preserve_folders: false,
    }
}

fn write_image_stack_output_for_run(
    image: &DynamicImage,
    output_path: &Path,
    output_format: ImageStackOutputFormat,
    export_settings: &ExportSettings,
    source_path: &str,
    degradation_ledger: &DegradationLedger,
) -> Result<(), String> {
    let transformed_image =
        if export_settings.resize.is_some() || export_settings.watermark.is_some() {
            Some(crate::export_processing::apply_export_resize_and_watermark(
                image.clone(),
                export_settings,
            )?)
        } else {
            None
        };
    let image = transformed_image.as_ref().unwrap_or(image);
    let dimensions = image.dimensions();
    if dimensions.0 == 0 || dimensions.1 == 0 {
        return Err("The image-stack result is empty and cannot be saved.".to_string());
    }

    let parent = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| {
        format!(
            "Failed to create the image-stack output folder '{}': {error}",
            parent.display()
        )
    })?;

    let mut temporary = NamedTempFile::new_in(parent).map_err(|error| {
        format!(
            "Failed to create a temporary image-stack file beside '{}': {error}",
            output_path.display()
        )
    })?;
    encode_image_stack_file(
        image,
        temporary.as_file_mut(),
        output_format,
        export_settings,
        source_path,
    )?;
    temporary.as_file().sync_all().map_err(|error| {
        format!(
            "Failed to sync the image-stack {} to disk: {error}",
            output_format.label()
        )
    })?;

    let temporary_size = temporary
        .as_file()
        .metadata()
        .map_err(|error| {
            format!(
                "Failed to inspect the saved image-stack {}: {error}",
                output_format.label()
            )
        })?
        .len();
    if temporary_size == 0 {
        return Err(format!(
            "The encoded image-stack {} is empty.",
            output_format.label()
        ));
    }

    validate_image_stack_output(
        temporary.path(),
        dimensions,
        output_format,
        crate::export_processing::effective_export_bit_depth(
            output_format.canonical_extension(),
            export_settings.bit_depth,
        ),
        image.color().has_alpha(),
        export_settings.embed_color_profile,
    )?;

    let temporary_path = temporary.into_temp_path();
    let mut file_system = StandardFinalOutputFileSystem;
    match DegradationManager::new(degradation_ledger).publish_staged_output(
        &mut file_system,
        temporary_path.as_ref(),
        output_path,
    )? {
        OutputPublication::Published(_) => Ok(()),
        OutputPublication::Rejected => Err(
            "The image-stack result was rejected; the diagnostic preview and Stack_Report were retained."
                .to_string(),
        ),
        OutputPublication::Cancelled => Err(
            "The image-stack export was cancelled; the diagnostic preview and Stack_Report were retained."
                .to_string(),
        ),
    }
}

fn write_image_stack_output_with_settings(
    image: &DynamicImage,
    output_path: &Path,
    output_format: ImageStackOutputFormat,
    export_settings: &ExportSettings,
    source_path: &str,
) -> Result<(), String> {
    write_image_stack_output_for_run(
        image,
        output_path,
        output_format,
        export_settings,
        source_path,
        &DegradationLedger::new(),
    )
}

#[cfg(test)]
fn write_image_stack_output(
    image: &DynamicImage,
    output_path: &Path,
    output_format: ImageStackOutputFormat,
) -> Result<(), String> {
    write_image_stack_output_with_settings(
        image,
        output_path,
        output_format,
        &default_image_stack_export_settings(),
        "",
    )
}

#[cfg(test)]
pub(crate) fn write_srgb_tiff(image: &DynamicImage, output_path: &Path) -> Result<(), String> {
    write_image_stack_output(image, output_path, ImageStackOutputFormat::Tiff)
}

#[cfg(test)]
pub(crate) fn write_srgb_jpeg(image: &DynamicImage, output_path: &Path) -> Result<(), String> {
    write_image_stack_output(image, output_path, ImageStackOutputFormat::Jpeg)
}

fn resolve_image_stack_output_path(
    first_path: &Path,
    blend_mode: &str,
    output_path_str: Option<&str>,
    output_format: ImageStackOutputFormat,
) -> Result<PathBuf, String> {
    if let Some(requested_path) = output_path_str.filter(|path| !path.trim().is_empty()) {
        let mut output_path = PathBuf::from(requested_path);
        let path_format = output_path
            .extension()
            .and_then(|value| value.to_str())
            .and_then(ImageStackOutputFormat::from_extension);
        if path_format != Some(output_format) {
            output_path.set_extension(output_format.canonical_extension());
        }
        return Ok(output_path);
    }

    let parent = first_path
        .parent()
        .ok_or_else(|| "Could not determine the source image folder.".to_string())?;
    let stem = first_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("image");
    let suffix = if blend_mode == "focus" {
        "FocusStack"
    } else {
        "Panorama"
    };
    Ok(parent.join(format!(
        "{stem}_{suffix}.{}",
        output_format.canonical_extension()
    )))
}

struct PreviewFiles {
    detail_height: u32,
    detail_path: String,
    detail_width: u32,
    interaction_height: u32,
    interaction_path: String,
    interaction_width: u32,
}

fn encode_preview_jpeg(image: &RgbImage, path: &Path, quality: u8) -> Result<(), String> {
    let file = File::create(path)
        .map_err(|error| format!("Failed to create image-stack preview: {error}"))?;
    let mut writer = BufWriter::new(file);
    let mut encoder = JpegEncoder::new_with_quality(&mut writer, quality);
    encoder
        .set_icc_profile(crate::color_management::srgb_v4_profile().to_vec())
        .map_err(|error| {
            format!("Failed to attach the image-stack preview color profile: {error}")
        })?;
    encoder
        .encode_image(image)
        .map_err(|error| format!("Failed to encode image-stack preview: {error}"))?;
    drop(encoder);
    writer
        .flush()
        .map_err(|error| format!("Failed to finish image-stack preview: {error}"))
}

/// The detail and interaction previews of a canonical stack result (需求 10.7):
/// both are downsampled from the result's own display-encoded pixels and from
/// nothing else, so a preview can never show a different image than the one
/// that is exported. `None` means the interaction preview is the detail one.
fn derive_preview_images(
    image: &DynamicImage,
    (detail_width, detail_height): (u32, u32),
    (interaction_width, interaction_height): (u32, u32),
) -> (RgbImage, Option<RgbImage>) {
    let (width, height) = image.dimensions();
    let full_rgb = image.to_rgb8();
    let detail_rgb = if (detail_width, detail_height) == (width, height) {
        full_rgb
    } else {
        image::imageops::resize(&full_rgb, detail_width, detail_height, FilterType::Lanczos3)
    };
    let interaction_rgb =
        ((interaction_width, interaction_height) != (detail_width, detail_height)).then(|| {
            image::imageops::resize(
                &detail_rgb,
                interaction_width,
                interaction_height,
                FilterType::Lanczos3,
            )
        });
    (detail_rgb, interaction_rgb)
}

/// Preview file names carry the result identifier, so the stored result and
/// its previews are linked by one identifier (需求 10.7).
fn preview_file_names(result_id: &str) -> (String, String) {
    (
        format!("{result_id}-detail.jpg"),
        format!("{result_id}-interaction.jpg"),
    )
}

fn write_preview_files(
    image: &DynamicImage,
    result_id: &str,
    app_handle: &AppHandle,
) -> Result<PreviewFiles, String> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err("The image result is empty.".to_string());
    }
    let (interaction_width, interaction_height) = preview_dimensions(width, height);
    let (detail_width, detail_height) = detail_preview_dimensions(width, height);
    let (detail_rgb, interaction_rgb) = derive_preview_images(
        image,
        (detail_width, detail_height),
        (interaction_width, interaction_height),
    );
    let (detail_name, interaction_name) = preview_file_names(result_id);

    let preview_dir = app_handle
        .path()
        .app_cache_dir()
        .map_err(|error| format!("Failed to resolve image-stack preview cache: {error}"))?
        .join("image-stack-previews");
    fs::create_dir_all(&preview_dir)
        .map_err(|error| format!("Failed to create image-stack preview cache: {error}"))?;
    let detail_path = preview_dir.join(detail_name);
    encode_preview_jpeg(&detail_rgb, &detail_path, DETAIL_PREVIEW_JPEG_QUALITY)?;

    let interaction_path = match interaction_rgb {
        None => detail_path.clone(),
        Some(interaction_rgb) => {
            let path = preview_dir.join(interaction_name);
            encode_preview_jpeg(&interaction_rgb, &path, PREVIEW_JPEG_QUALITY)?;
            path
        }
    };

    if let Ok(entries) = fs::read_dir(&preview_dir) {
        for entry in entries.flatten() {
            let stale_path = entry.path();
            if stale_path != interaction_path && stale_path != detail_path && stale_path.is_file() {
                let _ = fs::remove_file(stale_path);
            }
        }
    }

    Ok(PreviewFiles {
        detail_height,
        detail_path: detail_path.to_string_lossy().into_owned(),
        detail_width,
        interaction_height,
        interaction_path: interaction_path.to_string_lossy().into_owned(),
        interaction_width,
    })
}

#[tauri::command]
pub async fn process_image_stack(
    paths: Vec<String>,
    blend_mode: String,
    alignment_mode: String,
    pipeline_version: String,
    request_id: String,
    app_handle: AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    validate_image_stack_source_count(paths.len())?;
    if request_id.trim().is_empty() {
        return Err("Image-stack request ID is missing.".to_string());
    }
    validate_image_stack_pipeline_version(&pipeline_version)?;

    let source_paths: Vec<String> = paths
        .iter()
        .map(|path| parse_virtual_path(path).0.to_string_lossy().into_owned())
        .collect();
    let selected_blend_mode = resolve_blend_mode(&blend_mode);
    let selected_alignment_mode = AlignmentMode::from_wire(&alignment_mode);
    let result_handle = state.image_stack_result.clone();
    let generation_handle = state.image_stack_generation.clone();
    let generation = generation_handle.fetch_add(1, Ordering::SeqCst) + 1;
    crate::panorama_utils::stack_pipeline::degradation::set_active_run_generation(generation);
    *result_handle.lock().unwrap() = None;

    let task = tokio::task::spawn_blocking(move || {
        let result = stitch_images_with_options(
            source_paths,
            app_handle.clone(),
            selected_alignment_mode,
            selected_blend_mode,
            "image-stack-progress",
        );

        match result {
            Ok(outcome) => {
                if generation_handle.load(Ordering::SeqCst) != generation {
                    return Ok(());
                }

                let full_canvas_width = outcome.full_canvas_width;
                let full_canvas_height = outcome.full_canvas_height;
                let render_scale = outcome.render_scale;
                // The Coverage_Mask becomes alpha; the colour channels are the
                // opaque canonical ones either way (需求 10.6).
                let image = canonicalize_image_stack_result_with_coverage(
                    outcome.image,
                    outcome.coverage.as_ref(),
                );
                let _ = app_handle.emit("image-stack-progress", "Creating preview…");
                // One identifier names the stored result and its previews (需求 10.7).
                let result_id = Uuid::new_v4().to_string();
                let previews = write_preview_files(&image, &result_id, &app_handle)?;
                if generation_handle.load(Ordering::SeqCst) != generation {
                    let _ = fs::remove_file(&previews.interaction_path);
                    let _ = fs::remove_file(&previews.detail_path);
                    return Ok(());
                }

                // This is the one authoritative run-outcome decision. The
                // report has already folded the same ledger on the stitching
                // terminal path; carrying the snapshot with the image prevents
                // a later save from consulting another run's global ledger.
                let degradation_ledger =
                    crate::panorama_utils::stack_pipeline::degradation::run_ledger_snapshot();
                let outcome_decision = DegradationManager::new(&degradation_ledger).decision();
                if !outcome_decision.permits_final_output() {
                    // The composed pixels remain available only as diagnostic
                    // previews. They are intentionally not inserted into the
                    // exportable result store, and no cleanup targets either
                    // preview or the Stack_Report.
                    let message = match outcome_decision {
                        OutcomeDecision::Cancelled => {
                            "Image-stack processing was cancelled; diagnostic output was retained."
                        }
                        OutcomeDecision::Rejected => {
                            "The image-stack result was rejected; diagnostic output was retained."
                        }
                        OutcomeDecision::Success | OutcomeDecision::Degraded => unreachable!(),
                    }
                    .to_string();
                    let _ = app_handle.emit(
                        "image-stack-error",
                        serde_json::json!({
                            "message": message,
                            "requestId": request_id,
                            "result": outcome_decision.as_identifier(),
                            "previewPath": previews.interaction_path,
                            "detailPreviewPath": previews.detail_path,
                        }),
                    );
                    return Err(message);
                }

                let (source_width, source_height) = image.dimensions();
                {
                    let mut stored_result = result_handle.lock().unwrap();
                    if generation_handle.load(Ordering::SeqCst) != generation {
                        let _ = fs::remove_file(&previews.interaction_path);
                        let _ = fs::remove_file(&previews.detail_path);
                        return Ok(());
                    }
                    *stored_result = Some((result_id.clone(), image, degradation_ledger));
                }
                let _ = app_handle.emit(
                    "image-stack-complete",
                    serde_json::json!({
                        "previewPath": previews.interaction_path,
                        "previewWidth": previews.interaction_width,
                        "previewHeight": previews.interaction_height,
                        "detailPreviewPath": previews.detail_path,
                        "detailPreviewWidth": previews.detail_width,
                        "detailPreviewHeight": previews.detail_height,
                        "sourceWidth": source_width,
                        "sourceHeight": source_height,
                        "fullCanvasWidth": full_canvas_width,
                        "fullCanvasHeight": full_canvas_height,
                        "renderScale": render_scale,
                        "orderedPaths": outcome.ordered_paths,
                        "pipelineVersion": IMAGE_STACK_PIPELINE_VERSION,
                        "requestId": request_id,
                        "resultId": result_id,
                        "result": outcome_decision.as_identifier(),
                    }),
                );
                Ok(())
            }
            Err(error) => {
                if generation_handle.load(Ordering::SeqCst) == generation {
                    let _ = app_handle.emit(
                        "image-stack-error",
                        serde_json::json!({
                            "message": error,
                            "requestId": request_id,
                        }),
                    );
                }
                Err(error)
            }
        }
    });

    match task.await {
        Ok(result) => result,
        Err(error) => Err(format!("Image stack task failed: {error}")),
    }
}

#[cfg(test)]
mod tests {
    use std::fs::{self, File};
    use std::io::{BufReader, Cursor};
    use std::path::Path;

    use image::codecs::{jpeg::JpegDecoder, png::PngDecoder, tiff::TiffDecoder};
    use image::{
        ColorType, DynamicImage, GenericImageView, ImageBuffer, ImageDecoder, Rgb, Rgb32FImage,
    };

    use super::{
        DETAIL_PREVIEW_JPEG_QUALITY, FilterType, JpegEncoder, PREVIEW_JPEG_QUALITY, RgbImage,
        derive_preview_images, preview_file_names,
    };
    use super::{
        DETAIL_PREVIEW_MAX_LONG_SIDE, DETAIL_PREVIEW_MAX_PIXELS, IMAGE_STACK_MAX_SOURCES,
        IMAGE_STACK_PIPELINE_VERSION, ImageStackOutputFormat, PREVIEW_MAX_LONG_SIDE,
        PREVIEW_MAX_PIXELS, canonicalize_image_stack_result, default_image_stack_export_settings,
        detail_preview_dimensions, encode_srgb_tiff, preview_dimensions,
        resolve_image_stack_output_path, validate_image_stack_pipeline_version,
        validate_image_stack_source_count, write_image_stack_output,
        write_image_stack_output_for_run, write_image_stack_output_with_settings, write_srgb_tiff,
    };
    use crate::panorama_utils::stack_pipeline::degradation::{
        CLOSURE_UNRELIABLE_RESIDUAL, DegradationLedger, GEOMETRY_DISCONNECTED,
        RUN_CANCELLED_BY_USER,
    };

    #[test]
    fn image_stack_pipeline_version_rejects_stale_frontends() {
        assert!(validate_image_stack_pipeline_version(IMAGE_STACK_PIPELINE_VERSION).is_ok());
        assert!(validate_image_stack_pipeline_version("").is_err());
        assert!(validate_image_stack_pipeline_version("image-stack-legacy").is_err());
    }

    #[test]
    fn task_7_31_source_count_boundaries_are_exact() {
        let outcomes =
            [0usize, 1, 2, 500, 501].map(|count| (count, validate_image_stack_source_count(count)));
        assert!(outcomes[0].1.is_err(), "zero sources must be rejected");
        assert!(outcomes[1].1.is_err(), "one source must be rejected");
        assert!(outcomes[2].1.is_ok(), "two sources start the valid range");
        assert_eq!(IMAGE_STACK_MAX_SOURCES, 500);
        assert!(outcomes[3].1.is_ok(), "500 sources end the valid range");
        assert!(outcomes[4].1.is_err(), "501 sources exceed the valid range");
    }

    #[test]
    fn preview_dimensions_keep_images_within_the_quality_budget() {
        assert_eq!(preview_dimensions(4_000, 3_000), (4_000, 3_000));

        let (wide_width, wide_height) = preview_dimensions(20_000, 2_000);
        assert_eq!(wide_width, PREVIEW_MAX_LONG_SIDE);
        assert_eq!(wide_height, 410);

        let (large_width, large_height) = preview_dimensions(8_256, 5_504);
        assert!(large_width <= PREVIEW_MAX_LONG_SIDE);
        assert!(large_width as u64 * large_height as u64 <= PREVIEW_MAX_PIXELS + 10_000);

        let (portrait_width, portrait_height) = preview_dimensions(9_305, 12_618);
        assert!(portrait_height <= PREVIEW_MAX_LONG_SIDE);
        assert!(portrait_width as u64 * portrait_height as u64 <= PREVIEW_MAX_PIXELS + 10_000);

        let (detail_width, detail_height) = detail_preview_dimensions(9_305, 12_618);
        assert!(detail_height <= DETAIL_PREVIEW_MAX_LONG_SIDE);
        assert!(detail_width as u64 * detail_height as u64 <= DETAIL_PREVIEW_MAX_PIXELS + 10_000);
        assert!(detail_width > portrait_width);
        assert!(detail_height > portrait_height);
    }

    /// A painted-looking result: smooth colour fields with brush-scale detail.
    fn painted_result(width: u32, height: u32, seed: u64) -> DynamicImage {
        let phase = (seed % 997) as f32 * 0.013;
        let image = image::Rgb32FImage::from_fn(width, height, |x, y| {
            let (fx, fy) = (x as f32, y as f32);
            let broad = 0.45 + 0.25 * (fx * 0.021 + phase).sin() * (fy * 0.017 - phase).cos();
            let detail = 0.04 * ((fx * 0.9 + fy * 0.7 + phase * 5.0).sin());
            image::Rgb([
                (broad + detail).clamp(0.0, 1.0),
                (0.85 * broad + 0.05 + detail).clamp(0.0, 1.0),
                (0.6 * broad + 0.15 - detail).clamp(0.0, 1.0),
            ])
        });
        canonicalize_image_stack_result(DynamicImage::ImageRgb32F(image))
    }

    fn jpeg_round_trip(image: &RgbImage, quality: u8) -> RgbImage {
        let mut bytes = Vec::new();
        JpegEncoder::new_with_quality(&mut bytes, quality)
            .encode_image(image)
            .expect("in-memory preview JPEG");
        image::load_from_memory(&bytes)
            .expect("decode preview JPEG")
            .to_rgb8()
    }

    fn roi_mean(image: &RgbImage, (x, y, side): (u32, u32, u32)) -> [f32; 3] {
        let mut sum = [0.0f64; 3];
        for py in y..y + side {
            for px in x..x + side {
                for (channel, value) in image.get_pixel(px, py).0.iter().enumerate() {
                    sum[channel] += f64::from(*value) / 255.0;
                }
            }
        }
        let count = f64::from(side * side);
        sum.map(|value| (value / count) as f32)
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig {
            cases: 100,
            ..proptest::prelude::ProptestConfig::default()
        })]

        // Feature: layered-camera-group-focus-stitching, Property 58: 对于任意最终结果及其预览，
        // 两者携带同一个结果标识，预览的每个像素都可由最终结果的规范化显示编码像素降采样得到，
        // 且任意对应 ROI 的低频均值 Delta_E00 不超过 1.0。
        //
        // The previews are produced by `derive_preview_images` from the canonical result
        // alone, encoded at the production JPEG qualities, and named by the result
        // identifier. ROIs are aligned to the 2× and 4× downsampling so the preview ROI is
        // exactly the final ROI's footprint.
        //
        // **Validates: Requirements 10.7**
        #[test]
        fn property_58_previews_derive_from_the_result_with_bounded_roi_delta_e(
            seed in proptest::prelude::any::<u64>(),
            cells in 4u32..10,
            roi_x in 0u32..64,
            roi_y in 0u32..64,
        ) {
            let side = cells * 32;
            let result = painted_result(side, side, seed);
            let detail_dims = (side / 2, side / 2);
            let interaction_dims = (side / 4, side / 4);
            let (detail, interaction) = derive_preview_images(&result, detail_dims, interaction_dims);
            // Derived from the canonical pixels and from nothing else.
            let expected_detail = image::imageops::resize(
                &result.to_rgb8(),
                detail_dims.0,
                detail_dims.1,
                FilterType::Lanczos3,
            );
            proptest::prop_assert_eq!(&detail, &expected_detail);
            let interaction = interaction.expect("a smaller interaction preview");
            let (again_detail, again_interaction) =
                derive_preview_images(&result, detail_dims, interaction_dims);
            proptest::prop_assert_eq!(&detail, &again_detail);
            proptest::prop_assert_eq!(Some(&interaction), again_interaction.as_ref());

            let final_rgb = result.to_rgb8();
            let detail = jpeg_round_trip(&detail, DETAIL_PREVIEW_JPEG_QUALITY);
            let interaction = jpeg_round_trip(&interaction, PREVIEW_JPEG_QUALITY);
            // A 64 px final ROI anywhere on the 4-px lattice.
            let roi_side = 64u32;
            let x = (roi_x * 4).min(side - roi_side);
            let y = (roi_y * 4).min(side - roi_side);
            let reference = roi_mean(&final_rgb, (x, y, roi_side));
            for (preview, factor) in [(&detail, 2u32), (&interaction, 4u32)] {
                let mean = roi_mean(preview, (x / factor, y / factor, roi_side / factor));
                let delta = crate::panorama_utils::stack_pipeline::tone::delta_e00_rgb(reference, mean);
                proptest::prop_assert!(
                    delta <= 1.0,
                    "{factor}x preview ROI ({x}, {y}) Delta_E00 {delta}"
                );
            }

            // The stored result and both preview files carry one identifier.
            let result_id = format!("{seed:016x}");
            let (detail_name, interaction_name) = preview_file_names(&result_id);
            proptest::prop_assert!(detail_name.starts_with(&result_id));
            proptest::prop_assert!(interaction_name.starts_with(&result_id));
            proptest::prop_assert_ne!(detail_name, interaction_name);
        }
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig {
            cases: 100,
            ..proptest::prelude::ProptestConfig::default()
        })]

        // Feature: layered-camera-group-focus-stitching, Property 57: 对于任意最终结果，以支持
        // alpha 的格式写出时未覆盖像素的 alpha 为完全透明、已覆盖像素的 alpha 为完全不透明，
        // 且已覆盖像素的颜色通道值与不写入 alpha 时逐值相同。
        //
        // The canonical result with its Coverage_Mask is written as 16-bit TIFF and PNG and
        // read back; the colour channels are compared with the opaque canonical form.
        //
        // **Validates: Requirements 10.6**
        #[test]
        fn property_57_alpha_follows_coverage_without_changing_colour(
            seed in proptest::prelude::any::<u64>(),
            width in 4u32..40,
            height in 4u32..40,
        ) {
            let result = painted_result(width, height, seed);
            let coverage = image::GrayImage::from_fn(width, height, |x, y| {
                let hash = (u64::from(x) * 73_856_093) ^ (u64::from(y) * 19_349_663) ^ seed;
                image::Luma([if hash % 5 == 0 { 0 } else { 255 }])
            });
            let opaque = super::canonicalize_image_stack_result(result.clone()).to_rgb16();
            let with_alpha =
                super::canonicalize_image_stack_result_with_coverage(result, Some(&coverage));
            for format in [ImageStackOutputFormat::Tiff, ImageStackOutputFormat::Png] {
                let mut bytes = Cursor::new(Vec::new());
                super::encode_srgb_image_stack(&with_alpha, &mut bytes, format, 95, 16, true, None)
                    .expect("encode");
                let decoded = image::load_from_memory(bytes.get_ref()).expect("decode");
                proptest::prop_assert_eq!(decoded.color(), ColorType::Rgba16);
                let decoded = decoded.to_rgba16();
                for (x, y, pixel) in decoded.enumerate_pixels() {
                    let covered = coverage.get_pixel(x, y)[0] != 0;
                    proptest::prop_assert_eq!(pixel[3], if covered { u16::MAX } else { 0 });
                    proptest::prop_assert_eq!(&pixel.0[..3], &opaque.get_pixel(x, y).0[..]);
                }
            }
        }
    }

    #[test]
    fn output_fidelity_reports_bit_depth_and_alpha_downgrades_before_writing() {
        let coverage =
            image::GrayImage::from_fn(8, 8, |x, _| image::Luma([if x < 2 { 0 } else { 255 }]));
        let transparent = super::canonicalize_image_stack_result_with_coverage(
            painted_result(8, 8, 3),
            Some(&coverage),
        );
        let tiff = super::OutputFidelity::of(&transparent, ImageStackOutputFormat::Tiff, 16);
        assert_eq!(
            (tiff.bit_depth, tiff.has_transparency, tiff.alpha_preserved),
            (16, true, true)
        );
        assert!(tiff.downgrades(ImageStackOutputFormat::Tiff).is_empty());

        let jpeg = super::OutputFidelity::of(&transparent, ImageStackOutputFormat::Jpeg, 16);
        let notices = jpeg.downgrades(ImageStackOutputFormat::Jpeg);
        assert_eq!(notices.len(), 2, "{notices:?}");
        assert_eq!(notices[0].id, "output_bit_depth_downgraded");
        assert_eq!(notices[0].params.format, "JPEG");
        assert_eq!(notices[0].params.bit_depth, Some(8));
        assert_eq!(notices[1].id, "output_alpha_unsupported");
        assert_eq!(notices[1].params.format, "JPEG");
        assert_eq!(notices[1].params.bit_depth, None);

        // A fully covered result loses nothing to a missing alpha channel.
        let opaque = super::canonicalize_image_stack_result(painted_result(8, 8, 3));
        let jpeg = super::OutputFidelity::of(&opaque, ImageStackOutputFormat::Jpeg, 16);
        assert_eq!(jpeg.downgrades(ImageStackOutputFormat::Jpeg).len(), 1);
        let png8 = super::OutputFidelity::of(&transparent, ImageStackOutputFormat::Png, 8);
        assert_eq!(png8.downgrades(ImageStackOutputFormat::Png).len(), 1);
    }

    #[test]
    fn a_saved_tiff_with_alpha_validates_and_keeps_its_transparency() {
        let coverage = image::GrayImage::from_fn(16, 12, |x, y| {
            image::Luma([if (x + y) % 7 == 0 { 0 } else { 255 }])
        });
        let image = super::canonicalize_image_stack_result_with_coverage(
            painted_result(16, 12, 11),
            Some(&coverage),
        );
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("stack.tiff");
        write_image_stack_output(&image, &path, ImageStackOutputFormat::Tiff)
            .expect("the RGBA result is written and validated");
        let decoded = image::open(&path).expect("reopen").to_rgba16();
        for (x, y, pixel) in decoded.enumerate_pixels() {
            let covered = coverage.get_pixel(x, y)[0] != 0;
            assert_eq!(pixel[3], if covered { u16::MAX } else { 0 }, "({x}, {y})");
        }
    }

    #[test]
    fn canonical_stack_result_is_display_encoded_rgb16() {
        let source =
            DynamicImage::ImageRgb32F(Rgb32FImage::from_pixel(2, 1, Rgb([0.25, 0.5, 0.75])));
        let canonical = canonicalize_image_stack_result(source);
        let pixels = canonical.as_rgb16().expect("canonical RGB16 result");

        assert_eq!(pixels.dimensions(), (2, 1));
        assert!((pixels.get_pixel(0, 0)[0] as i32 - 16_384).abs() <= 1);
        assert!((pixels.get_pixel(0, 0)[1] as i32 - 32_768).abs() <= 1);
        assert!((pixels.get_pixel(0, 0)[2] as i32 - 49_151).abs() <= 1);
    }

    #[test]
    fn exported_tiff_preserves_canonical_pixels_and_srgb_profile() {
        let source = DynamicImage::ImageRgb32F(Rgb32FImage::from_fn(3, 2, |x, y| {
            Rgb([
                (x as f32 + 1.0) / 4.0,
                (y as f32 + 1.0) / 3.0,
                (x as f32 + y as f32 + 1.0) / 6.0,
            ])
        }));
        let canonical = canonicalize_image_stack_result(source);
        let expected = canonical.to_rgb16();
        let mut encoded = Cursor::new(Vec::new());
        encode_srgb_tiff(&canonical, &mut encoded).expect("encode canonical TIFF");

        encoded.set_position(0);
        let mut metadata_decoder = TiffDecoder::new(&mut encoded).expect("decode TIFF metadata");
        let profile = metadata_decoder
            .icc_profile()
            .expect("read TIFF ICC")
            .expect("embedded TIFF ICC");
        assert_eq!(profile, crate::color_management::srgb_v4_profile());

        encoded.set_position(0);
        let decoded = DynamicImage::from_decoder(
            TiffDecoder::new(&mut encoded).expect("decode canonical TIFF"),
        )
        .expect("read canonical TIFF pixels")
        .to_rgb16();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn atomic_tiff_save_replaces_stale_output_with_a_valid_image() {
        let directory = tempfile::tempdir().expect("temporary image-stack output directory");
        let output_path = directory.path().join("stack-result.tiff");
        fs::write(&output_path, b"stale").expect("seed stale output");
        let source = DynamicImage::ImageRgb32F(Rgb32FImage::from_fn(4, 3, |x, y| {
            Rgb([
                (x as f32 + 1.0) / 5.0,
                (y as f32 + 1.0) / 4.0,
                (x as f32 + y as f32 + 1.0) / 8.0,
            ])
        }));
        let canonical = canonicalize_image_stack_result(source);

        write_srgb_tiff(&canonical, &output_path).expect("atomic image-stack TIFF save");
        assert!(
            fs::metadata(&output_path)
                .expect("saved TIFF metadata")
                .len()
                > 5
        );

        let mut decoder = TiffDecoder::new(BufReader::new(
            File::open(&output_path).expect("open persisted image-stack TIFF"),
        ))
        .expect("decode persisted image-stack TIFF");
        assert_eq!(decoder.dimensions(), (4, 3));
        assert_eq!(
            decoder
                .icc_profile()
                .expect("read persisted TIFF ICC")
                .as_deref(),
            Some(crate::color_management::srgb_v4_profile())
        );
    }

    #[test]
    fn rejected_stack_save_preserves_existing_outputs_and_removes_only_its_temp() {
        let directory = tempfile::tempdir().expect("temporary rejected stack directory");
        let output_path = directory.path().join("stack-result.tiff");
        let report_path = directory.path().join("stack-report.json");
        let preview_path = directory.path().join("diagnostic-preview.jpg");
        fs::write(&output_path, b"pre-existing final").expect("seed existing final");
        fs::write(&report_path, b"stack report").expect("seed report");
        fs::write(&preview_path, b"diagnostic preview").expect("seed preview");

        let source = DynamicImage::ImageRgb16(ImageBuffer::from_pixel(4, 3, Rgb([1, 2, 3])));
        let mut ledger = DegradationLedger::new();
        ledger.record(CLOSURE_UNRELIABLE_RESIDUAL, serde_json::Value::Null);
        ledger.record(GEOMETRY_DISCONNECTED, serde_json::Value::Null);

        let error = write_image_stack_output_for_run(
            &source,
            &output_path,
            ImageStackOutputFormat::Tiff,
            &default_image_stack_export_settings(),
            "",
            &ledger,
        )
        .expect_err("a rejected run must not publish its staged result");

        assert!(error.contains("rejected"));
        assert_eq!(
            fs::read(&output_path).expect("read existing final"),
            b"pre-existing final"
        );
        assert_eq!(
            fs::read(&report_path).expect("read report"),
            b"stack report"
        );
        assert_eq!(
            fs::read(&preview_path).expect("read preview"),
            b"diagnostic preview"
        );
        let entries = fs::read_dir(directory.path())
            .expect("list rejected stack directory")
            .map(|entry| entry.expect("directory entry").file_name())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            entries,
            [
                "diagnostic-preview.jpg",
                "stack-report.json",
                "stack-result.tiff"
            ]
            .into_iter()
            .map(std::ffi::OsString::from)
            .collect(),
            "the rejected run must remove only its generated temporary file"
        );
    }

    #[test]
    fn cancelled_stack_save_preserves_existing_final_and_removes_only_its_temp() {
        let directory = tempfile::tempdir().expect("temporary cancelled stack directory");
        let output_path = directory.path().join("stack-result.tiff");
        let user_path = directory.path().join("user-notes.txt");
        fs::write(&output_path, b"pre-existing final").expect("seed existing final");
        fs::write(&user_path, b"keep me").expect("seed user file");

        let source = DynamicImage::ImageRgb16(ImageBuffer::from_pixel(4, 3, Rgb([1, 2, 3])));
        let mut ledger = DegradationLedger::new();
        ledger.record(GEOMETRY_DISCONNECTED, serde_json::Value::Null);
        ledger.record(RUN_CANCELLED_BY_USER, serde_json::Value::Null);

        let error = write_image_stack_output_for_run(
            &source,
            &output_path,
            ImageStackOutputFormat::Tiff,
            &default_image_stack_export_settings(),
            "",
            &ledger,
        )
        .expect_err("a cancelled run must not publish its staged result");

        assert!(error.contains("cancelled"));
        assert_eq!(
            fs::read(&output_path).expect("read existing final"),
            b"pre-existing final"
        );
        assert_eq!(fs::read(&user_path).expect("read user file"), b"keep me");
        let entries = fs::read_dir(directory.path())
            .expect("list cancelled stack directory")
            .map(|entry| entry.expect("directory entry").file_name())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            entries,
            ["stack-result.tiff", "user-notes.txt"]
                .into_iter()
                .map(std::ffi::OsString::from)
                .collect(),
            "the cancelled run must remove only its generated temporary file"
        );
    }

    #[test]
    fn save_path_prefers_the_requested_destination_and_normalizes_the_extension() {
        let source_path = Path::new("/photos/source.jpg");
        assert_eq!(
            resolve_image_stack_output_path(
                source_path,
                "focus",
                Some("/exports/custom-stack"),
                ImageStackOutputFormat::Png,
            )
            .expect("explicit PNG save destination"),
            Path::new("/exports/custom-stack.png")
        );
        assert_eq!(
            resolve_image_stack_output_path(
                source_path,
                "focus",
                Some("/exports/custom-stack.jpeg"),
                ImageStackOutputFormat::Jpeg,
            )
            .expect("explicit JPEG save destination"),
            Path::new("/exports/custom-stack.jpeg")
        );
        assert_eq!(
            resolve_image_stack_output_path(
                source_path,
                "focus",
                Some("/exports/custom-stack.png"),
                ImageStackOutputFormat::Tiff,
            )
            .expect("selected format overrides a mismatched extension"),
            Path::new("/exports/custom-stack.tiff")
        );
        assert_eq!(
            resolve_image_stack_output_path(
                source_path,
                "focus",
                None,
                ImageStackOutputFormat::Tiff,
            )
            .expect("fallback TIFF save destination"),
            Path::new("/photos/source_FocusStack.tiff")
        );
    }

    #[test]
    fn output_format_accepts_supported_aliases_and_rejects_other_values() {
        assert_eq!(
            ImageStackOutputFormat::from_wire("tif"),
            Ok(ImageStackOutputFormat::Tiff)
        );
        assert_eq!(
            ImageStackOutputFormat::from_wire("JPEG"),
            Ok(ImageStackOutputFormat::Jpeg)
        );
        assert!(ImageStackOutputFormat::from_wire("webp").is_err());
    }

    #[test]
    fn atomic_stack_save_supports_tiff_png_and_jpeg() {
        let directory = tempfile::tempdir().expect("temporary image-stack output directory");
        let source = DynamicImage::ImageRgb16(ImageBuffer::from_fn(8, 5, |x, y| {
            Rgb([
                ((x + 1) * 4_096) as u16,
                ((y + 1) * 8_192) as u16,
                ((x + y + 1) * 2_048) as u16,
            ])
        }));
        let expected_lossless_pixels = source.to_rgb16();

        for (format, extension) in [
            (ImageStackOutputFormat::Tiff, "tiff"),
            (ImageStackOutputFormat::Png, "png"),
            (ImageStackOutputFormat::Jpeg, "jpg"),
        ] {
            let output_path = directory.path().join(format!("stack-result.{extension}"));
            fs::write(&output_path, b"stale").expect("seed stale output");

            write_image_stack_output(&source, &output_path, format)
                .expect("atomic multi-format image-stack save");

            assert!(
                fs::metadata(&output_path)
                    .expect("saved output metadata")
                    .len()
                    > 5
            );
            let decoded = image::open(&output_path).expect("decode saved multi-format output");
            assert_eq!(decoded.dimensions(), source.dimensions());
            if format != ImageStackOutputFormat::Jpeg {
                assert_eq!(decoded.to_rgb16(), expected_lossless_pixels);
            }
        }
    }

    #[test]
    fn stack_export_respects_selected_bit_depth_for_lossless_formats() {
        let directory = tempfile::tempdir().expect("temporary image-stack output directory");
        let source = DynamicImage::ImageRgb16(ImageBuffer::from_fn(8, 5, |x, y| {
            Rgb([
                ((x + 1) * 4_096) as u16,
                ((y + 1) * 8_192) as u16,
                ((x + y + 1) * 2_048) as u16,
            ])
        }));

        for bit_depth in [8_u8, 16_u8] {
            for (format, extension) in [
                (ImageStackOutputFormat::Tiff, "tiff"),
                (ImageStackOutputFormat::Png, "png"),
            ] {
                let output_path = directory
                    .path()
                    .join(format!("stack-result-{bit_depth}.{extension}"));
                let mut settings = default_image_stack_export_settings();
                settings.bit_depth = bit_depth;
                write_image_stack_output_with_settings(
                    &source,
                    &output_path,
                    format,
                    &settings,
                    "",
                )
                .expect("save selected image-stack bit depth");

                let actual_color_type = match format {
                    ImageStackOutputFormat::Tiff => TiffDecoder::new(BufReader::new(
                        File::open(&output_path).expect("open saved TIFF"),
                    ))
                    .expect("decode saved TIFF")
                    .color_type(),
                    ImageStackOutputFormat::Png => PngDecoder::new(BufReader::new(
                        File::open(&output_path).expect("open saved PNG"),
                    ))
                    .expect("decode saved PNG")
                    .color_type(),
                    ImageStackOutputFormat::Jpeg => unreachable!(),
                };
                assert_eq!(
                    actual_color_type,
                    if bit_depth == 16 {
                        ColorType::Rgb16
                    } else {
                        ColorType::Rgb8
                    }
                );
            }
        }
    }

    #[test]
    fn shared_export_settings_resize_stack_output_and_write_selected_metadata() {
        fn ascii_tag(exif_data: &exif::Exif, tag: exif::Tag) -> Option<String> {
            let field = exif_data.get_field(tag, exif::In::PRIMARY)?;
            let exif::Value::Ascii(values) = &field.value else {
                return None;
            };
            Some(
                values
                    .iter()
                    .map(|value| {
                        String::from_utf8_lossy(value)
                            .trim_end_matches('\0')
                            .to_string()
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
            )
        }

        let directory = tempfile::tempdir().expect("temporary shared export directory");
        let output_path = directory.path().join("stack-settings.jpg");
        let source = DynamicImage::ImageRgb16(ImageBuffer::from_fn(8, 5, |x, y| {
            Rgb([
                ((x + 1) * 4_096) as u16,
                ((y + 1) * 8_192) as u16,
                ((x + y + 1) * 2_048) as u16,
            ])
        }));
        let settings = crate::export_processing::ExportSettings {
            bit_depth: 8,
            jpeg_quality: 37,
            resize: Some(crate::export_processing::ResizeOptions {
                mode: crate::export_processing::ResizeMode::Width,
                value: 4,
                dont_enlarge: false,
            }),
            keep_metadata: false,
            metadata_overrides: Some(crate::exif_processing::ExportMetadataOverrides {
                artist: Some("Museum Imaging Team".to_string()),
                contact: Some("archive@example.test".to_string()),
                copyright: Some("Copyright 2026".to_string()),
                description: Some("Focus-stacked artifact".to_string()),
            }),
            preserve_timestamps: false,
            strip_gps: true,
            embed_color_profile: false,
            filename_template: None,
            watermark: None,
            export_masks: false,
            preserve_folders: false,
        };

        write_image_stack_output_with_settings(
            &source,
            &output_path,
            ImageStackOutputFormat::Jpeg,
            &settings,
            "/missing/source.jpg",
        )
        .expect("save stack through shared export settings");

        let encoded = fs::read(&output_path).expect("read shared export output");
        let mut decoder =
            JpegDecoder::new(Cursor::new(&encoded)).expect("decode resized stack JPEG");
        assert_eq!(decoder.dimensions(), (4, 3));
        assert_eq!(
            decoder.icc_profile().expect("read optional ICC profile"),
            None
        );

        let exif_data = exif::Reader::new()
            .read_from_container(&mut Cursor::new(&encoded))
            .expect("read selected stack metadata");
        assert_eq!(
            ascii_tag(&exif_data, exif::Tag::Artist).as_deref(),
            Some("Museum Imaging Team")
        );
        assert_eq!(
            ascii_tag(&exif_data, exif::Tag::Copyright).as_deref(),
            Some("Copyright 2026")
        );
        assert_eq!(
            ascii_tag(&exif_data, exif::Tag::ImageDescription).as_deref(),
            Some("Focus-stacked artifact")
        );
        let contact = exif_data
            .get_field(exif::Tag::UserComment, exif::In::PRIMARY)
            .and_then(|field| match &field.value {
                exif::Value::Undefined(value, _) => {
                    Some(String::from_utf8_lossy(value).to_string())
                }
                _ => None,
            });
        assert_eq!(contact.as_deref(), Some("Contact: archive@example.test"));
    }

    #[test]
    #[ignore = "requires RAW_EDITOR_STACK_EXPORT_SOURCE and RAW_EDITOR_STACK_EXPORT_OUTPUT_DIR"]
    fn real_stack_export_fixture_from_env() {
        let source_path = std::env::var_os("RAW_EDITOR_STACK_EXPORT_SOURCE")
            .map(std::path::PathBuf::from)
            .expect("RAW_EDITOR_STACK_EXPORT_SOURCE must point to a rendered stack image");
        let output_dir = std::env::var_os("RAW_EDITOR_STACK_EXPORT_OUTPUT_DIR")
            .map(std::path::PathBuf::from)
            .expect("RAW_EDITOR_STACK_EXPORT_OUTPUT_DIR must point to a writable directory");
        fs::create_dir_all(&output_dir).expect("create real image-stack export output directory");

        let mut reader = image::ImageReader::open(&source_path)
            .expect("open real image-stack export fixture")
            .with_guessed_format()
            .expect("detect real image-stack export fixture format");
        reader.no_limits();
        let decoded = reader
            .decode()
            .expect("decode real image-stack export fixture");
        let canonical = if decoded.as_rgb16().is_some() {
            decoded
        } else {
            canonicalize_image_stack_result(decoded)
        };

        for (format, extension) in [
            (ImageStackOutputFormat::Tiff, "tiff"),
            (ImageStackOutputFormat::Png, "png"),
            (ImageStackOutputFormat::Jpeg, "jpg"),
        ] {
            let output_path = output_dir.join(format!("real-stack-export.{extension}"));
            write_image_stack_output(&canonical, &output_path, format)
                .expect("export real image-stack fixture");
            println!("{}", output_path.display());
        }
    }
}

/// 需求 10.10: the notices of saving the current result in `output_format`
/// with `bit_depth` (fewer than 16 bits per channel, or transparency the
/// format cannot keep), returned before anything is written so the caller can
/// show them ahead of `save_image_stack`.
#[tauri::command]
pub async fn image_stack_output_notices(
    output_format: String,
    bit_depth: u8,
    result_id: String,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<OutputDowngradeNotice>, String> {
    let output_format = ImageStackOutputFormat::from_wire(&output_format)?;
    let result = state
        .image_stack_result
        .lock()
        .map_err(|_| "The image-stack result store is unavailable.".to_string())?;
    let (stored_result_id, image, _) = result
        .as_ref()
        .ok_or_else(|| "No image-stack result is available to save.".to_string())?;
    if stored_result_id != &result_id {
        return Err(
            "The visible image-stack preview is no longer the current result. Please realign before saving."
                .to_string(),
        );
    }
    Ok(OutputFidelity::of(image, output_format, bit_depth).downgrades(output_format))
}

#[tauri::command]
pub async fn save_image_stack(
    first_path_str: String,
    blend_mode: String,
    output_format: String,
    export_settings: ExportSettings,
    result_id: String,
    output_path_str: Option<String>,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let (first_path, _) = parse_virtual_path(&first_path_str);
    let output_format = ImageStackOutputFormat::from_wire(&output_format)?;
    let output_path = resolve_image_stack_output_path(
        &first_path,
        &blend_mode,
        output_path_str.as_deref(),
        output_format,
    )?;
    let output_path_for_task = output_path.clone();
    let sidecar_source = first_path.to_string_lossy().into_owned();
    let result_handle = state.image_stack_result.clone();

    tokio::task::spawn_blocking(move || {
        let result = result_handle
            .lock()
            .map_err(|_| "The image-stack result store is unavailable.".to_string())?;
        let (stored_result_id, image, degradation_ledger) = result
            .as_ref()
            .ok_or_else(|| "No image-stack result is available to save.".to_string())?;
        if stored_result_id != &result_id {
            return Err(
                "The visible image-stack preview is no longer the current result. Please realign before saving."
                    .to_string(),
            );
        }

        write_image_stack_output_for_run(
            image,
            &output_path_for_task,
            output_format,
            &export_settings,
            &sidecar_source,
            degradation_ledger,
        )?;
        drop(result);
        // Keep the canonical result cached so the same stack can be exported again in
        // another format or to another destination without running the expensive alignment
        // pass a second time. The encoded file is the export contract. Copying the source
        // sidecar here would silently restore EXIF/GPS fields that the user explicitly
        // removed in the dialog.
        Ok(output_path_for_task.to_string_lossy().into_owned())
    })
    .await
    .map_err(|error| format!("Image-stack save task failed: {error}"))?
}
