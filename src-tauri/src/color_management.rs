use std::fs::File;
use std::io::{BufWriter, Cursor};
use std::path::Path;
use std::sync::OnceLock;

use base64::{Engine as _, engine::general_purpose};
use image::codecs::{jpeg::JpegEncoder, png::PngEncoder, tiff::TiffEncoder};
use image::{DynamicImage, ImageEncoder, ImageFormat};
use moxcms::{
    CicpColorPrimaries, CicpProfile, ColorProfile, DataColorSpace, Layout, MatrixCoefficients,
    ParsingOptions, TransferCharacteristics, TransformOptions,
};
use mozjpeg_rs::{Encoder, Preset};

/// Compact ICC Profiles' CC0 sRGB v4 profile.
/// Source: https://github.com/saucecontrol/Compact-ICC-Profiles/blob/master/profiles/sRGB-v4.icc
const SRGB_V4_BASE64: &str = "AAAB4GxjbXMEIAAAbW50clJHQiBYWVogB+IAAwAUAAkADgAdYWNzcE1TRlQAAAAAc2F3c2N0cmwAAAAAAAAAAAAAAAAAAPbWAAEAAAAA0y1oYW5keem/Vlo+AbaDI4VVRvdPqgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKZGVzYwAAAPwAAAAkY3BydAAAASAAAAAid3RwdAAAAUQAAAAUY2hhZAAAAVgAAAAsclhZWgAAAYQAAAAUZ1hZWgAAAZgAAAAUYlhZWgAAAawAAAAUclRSQwAAAcAAAAAgZ1RSQwAAAcAAAAAgYlRSQwAAAcAAAAAgbWx1YwAAAAAAAAABAAAADGVuVVMAAAAIAAAAHABzAFIARwBCbWx1YwAAAAAAAAABAAAADGVuVVMAAAAGAAAAHABDAEMAMAAAWFlaIAAAAAAAAPbWAAEAAAAA0y1zZjMyAAAAAAABDD8AAAXd///zJgAAB5AAAP2S///7of///aIAAAPcAADAcVhZWiAAAAAAAABvoAAAOPIAAAOPWFlaIAAAAAAAAGKWAAC3iQAAGNpYWVogAAAAAAAAJKAAAA+FAAC2xHBhcmEAAAAAAAMAAAACZmkAAPKnAAANWQAAE9AAAApb";

static SRGB_V4_PROFILE: OnceLock<Vec<u8>> = OnceLock::new();

pub const MAX_EMBEDDED_ICC_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IccColorSpace {
    Rgb,
    Gray,
    Cmyk,
    Other([u8; 4]),
}

/// IEC 61966-2-1 sRGB electro-optical transfer function for one encoded channel.
///
/// The editor's CPU and WGSL paths both use this contract. Values below zero are
/// clipped because the current preview/export target is display-referred sRGB.
#[inline]
pub fn srgb_to_linear_channel(encoded: f32) -> f32 {
    let encoded = encoded.max(0.0);
    if encoded <= 0.04045 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

/// IEC 61966-2-1 sRGB opto-electronic transfer function for one linear channel.
///
/// Positive values above 1.0 remain extended until the final 8-bit output clamp,
/// so highlight/tone-mapping stages can retain headroom before encoding.
#[inline]
pub fn linear_to_srgb_channel(linear: f32) -> f32 {
    let linear = linear.max(0.0);
    if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

pub fn inspect_icc_profile_header(profile: &[u8]) -> Result<IccColorSpace, String> {
    if profile.len() > MAX_EMBEDDED_ICC_BYTES {
        return Err(format!(
            "embedded ICC profile is {} bytes; limit is {MAX_EMBEDDED_ICC_BYTES}",
            profile.len()
        ));
    }
    if profile.len() < 128 {
        return Err("ICC profile is shorter than its 128-byte header".to_string());
    }

    let declared_size = u32::from_be_bytes(
        profile[0..4]
            .try_into()
            .map_err(|_| "ICC profile size field is missing".to_string())?,
    ) as usize;
    if declared_size != profile.len() {
        return Err(format!(
            "ICC profile declares {declared_size} bytes but contains {}",
            profile.len()
        ));
    }
    if &profile[36..40] != b"acsp" {
        return Err("ICC profile signature is not acsp".to_string());
    }

    let space: [u8; 4] = profile[16..20]
        .try_into()
        .map_err(|_| "ICC profile color space field is missing".to_string())?;
    Ok(match &space {
        b"RGB " => IccColorSpace::Rgb,
        b"GRAY" => IccColorSpace::Gray,
        b"CMYK" => IccColorSpace::Cmyk,
        _ => IccColorSpace::Other(space),
    })
}

pub fn validate_icc_profile(profile: &[u8]) -> Result<(), String> {
    match inspect_icc_profile_header(profile)? {
        IccColorSpace::Rgb => Ok(()),
        IccColorSpace::Gray => Err("ICC profile uses Gray color space, not RGB".to_string()),
        IccColorSpace::Cmyk => Err("ICC profile uses CMYK color space, not RGB".to_string()),
        IccColorSpace::Other(tag) => Err(format!(
            "ICC profile is not an RGB profile (space {:?})",
            String::from_utf8_lossy(&tag)
        )),
    }
}

pub fn srgb_v4_profile() -> &'static [u8] {
    SRGB_V4_PROFILE
        .get_or_init(|| {
            let profile = general_purpose::STANDARD
                .decode(SRGB_V4_BASE64)
                .expect("bundled sRGB v4 ICC profile must be valid Base64");
            validate_icc_profile(&profile).expect("bundled sRGB v4 ICC profile must be valid");
            profile
        })
        .as_slice()
}

fn bounded_icc_parsing_options() -> ParsingOptions {
    ParsingOptions {
        max_profile_size: MAX_EMBEDDED_ICC_BYTES + 1,
        max_allowed_clut_size: MAX_EMBEDDED_ICC_BYTES,
        max_allowed_trc_size: 65_536,
    }
}

fn apply_rgb_profile_transform_to_srgb(
    pixels: &mut [f32],
    source_profile: &ColorProfile,
) -> Result<(), String> {
    let destination_profile = ColorProfile::new_srgb();
    let transform_options = TransformOptions {
        rendering_intent: source_profile.rendering_intent,
        ..TransformOptions::default()
    };

    match source_profile.create_in_place_transform_f32(
        Layout::Rgb,
        &destination_profile,
        transform_options,
    ) {
        Ok(transform) => transform
            .transform(pixels)
            .map_err(|error| format!("failed to apply embedded ICC profile: {error}"))?,
        Err(_) => {
            let transform = source_profile
                .create_transform_f32(
                    Layout::Rgb,
                    &destination_profile,
                    Layout::Rgb,
                    transform_options,
                )
                .map_err(|error| format!("failed to create embedded ICC transform: {error}"))?;
            let mut converted = vec![0.0; pixels.len()];
            transform
                .transform(pixels, &mut converted)
                .map_err(|error| format!("failed to apply embedded ICC profile: {error}"))?;
            pixels.copy_from_slice(&converted);
        }
    }

    pixels
        .iter_mut()
        .for_each(|channel| *channel = channel.clamp(0.0, 1.0));
    Ok(())
}

/// Converts encoded RGB samples from an embedded ICC profile into encoded sRGB.
///
/// Common matrix-shaper profiles are transformed in place. LUT profiles fall back
/// to one temporary output buffer, keeping the usual photo path allocation-free.
/// The current editor contract is SDR sRGB, so out-of-gamut extended results are
/// clipped at this input boundary before the main shader applies the sRGB EOTF.
pub fn normalize_encoded_rgb_profile_to_srgb(
    pixels: &mut [f32],
    profile: &[u8],
) -> Result<bool, String> {
    if !pixels.len().is_multiple_of(3) {
        return Err("encoded RGB buffer does not contain complete pixels".to_string());
    }
    if profile.len() > MAX_EMBEDDED_ICC_BYTES {
        return Err(format!(
            "embedded ICC profile is {} bytes; limit is {MAX_EMBEDDED_ICC_BYTES}",
            profile.len()
        ));
    }

    validate_icc_profile(profile)?;
    if profile == srgb_v4_profile() || pixels.is_empty() {
        return Ok(false);
    }

    let source_profile =
        ColorProfile::new_from_slice_with_options(profile, bounded_icc_parsing_options())
            .map_err(|error| format!("failed to parse embedded ICC profile: {error}"))?;
    if source_profile.color_space != DataColorSpace::Rgb {
        return Err("embedded ICC profile is not an RGB profile".to_string());
    }

    let destination_profile = ColorProfile::new_srgb();
    let transform_options = TransformOptions {
        rendering_intent: source_profile.rendering_intent,
        ..TransformOptions::default()
    };

    match source_profile.create_in_place_transform_f32(
        Layout::Rgb,
        &destination_profile,
        transform_options,
    ) {
        Ok(transform) => transform
            .transform(pixels)
            .map_err(|error| format!("failed to apply embedded ICC profile: {error}"))?,
        Err(_) => {
            let transform = source_profile
                .create_transform_f32(
                    Layout::Rgb,
                    &destination_profile,
                    Layout::Rgb,
                    transform_options,
                )
                .map_err(|error| format!("failed to create embedded ICC transform: {error}"))?;
            let mut converted = vec![0.0; pixels.len()];
            transform
                .transform(pixels, &mut converted)
                .map_err(|error| format!("failed to apply embedded ICC profile: {error}"))?;
            pixels.copy_from_slice(&converted);
        }
    }

    pixels
        .iter_mut()
        .for_each(|channel| *channel = channel.clamp(0.0, 1.0));
    Ok(true)
}

/// Converts grayscale samples (stored as replicated `[g, g, g]` encoded floats)
/// from an embedded Gray ICC profile into encoded sRGB.
///
/// Non-Gray, malformed, or oversized profiles fail without mutating `pixels`.
pub fn normalize_encoded_gray_profile_to_srgb(
    pixels: &mut [f32],
    profile: &[u8],
) -> Result<bool, String> {
    if !pixels.len().is_multiple_of(3) {
        return Err("encoded RGB buffer does not contain complete pixels".to_string());
    }
    match inspect_icc_profile_header(profile)? {
        IccColorSpace::Gray => {}
        IccColorSpace::Rgb => {
            return Err("embedded ICC profile uses RGB color space, expected Gray".to_string());
        }
        IccColorSpace::Cmyk => {
            return Err("embedded ICC profile uses CMYK color space, expected Gray".to_string());
        }
        IccColorSpace::Other(tag) => {
            return Err(format!(
                "embedded ICC profile uses unsupported space {:?}, expected Gray",
                String::from_utf8_lossy(&tag)
            ));
        }
    }
    if pixels.is_empty() {
        return Ok(false);
    }

    let source_profile =
        ColorProfile::new_from_slice_with_options(profile, bounded_icc_parsing_options())
            .map_err(|error| format!("failed to parse embedded Gray ICC profile: {error}"))?;
    if source_profile.color_space != DataColorSpace::Gray {
        return Err("embedded ICC profile is not a Gray profile".to_string());
    }

    let destination_profile = ColorProfile::new_srgb();
    let transform_options = TransformOptions {
        rendering_intent: source_profile.rendering_intent,
        ..TransformOptions::default()
    };
    let transform = source_profile
        .create_transform_f32(
            Layout::Gray,
            &destination_profile,
            Layout::Rgb,
            transform_options,
        )
        .map_err(|error| format!("failed to create embedded Gray ICC transform: {error}"))?;

    let gray: Vec<f32> = pixels.chunks_exact(3).map(|rgb| rgb[0]).collect();
    let mut converted = vec![0.0_f32; pixels.len()];
    transform
        .transform(&gray, &mut converted)
        .map_err(|error| format!("failed to apply embedded Gray ICC profile: {error}"))?;

    for (dst, src) in pixels.iter_mut().zip(converted) {
        *dst = src.clamp(0.0, 1.0);
    }
    Ok(true)
}

/// Extracts a 4-byte PNG `cICP` (Coding-Independent Code Points) chunk payload
/// when present before `IDAT`/`IEND`. Returns `None` for non-PNG or malformed data.
pub fn extract_png_cicp_chunk(bytes: &[u8]) -> Option<[u8; 4]> {
    const PNG_SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
    if bytes.len() < PNG_SIGNATURE.len() || &bytes[..8] != PNG_SIGNATURE {
        return None;
    }

    let mut offset = 8usize;
    while offset.checked_add(12)? <= bytes.len() {
        let length = u32::from_be_bytes(bytes[offset..offset + 4].try_into().ok()?) as usize;
        let chunk_type = &bytes[offset + 4..offset + 8];
        let data_start = offset + 8;
        let data_end = data_start.checked_add(length)?;
        let chunk_end = data_end.checked_add(4)?;
        if chunk_end > bytes.len() {
            return None;
        }

        if chunk_type == b"cICP" {
            if length == 4 {
                return bytes[data_start..data_end].try_into().ok();
            }
            return None;
        }
        if chunk_type == b"IDAT" || chunk_type == b"IEND" {
            return None;
        }
        offset = chunk_end;
    }
    None
}

/// Converts encoded RGB pixels described solely by an ITU-T H.273 CICP 4-tuple
/// `[color_primaries, transfer_characteristics, matrix_coefficients, video_full_range_flag]`
/// into encoded sRGB.
///
/// Only full-range Identity-matrix SDR RGB CICP combinations are converted.
/// HDR (PQ/HLG), narrow-range, YCbCr, and reserved/unspecified tuples fail closed
/// without mutating `pixels`.
pub fn normalize_encoded_rgb_cicp_to_srgb(
    pixels: &mut [f32],
    cicp: [u8; 4],
) -> Result<bool, String> {
    if !pixels.len().is_multiple_of(3) {
        return Err("encoded RGB buffer does not contain complete pixels".to_string());
    }

    let [primaries_code, transfer_code, matrix_code, full_range_code] = cicp;
    if matrix_code != 0 {
        return Err(format!(
            "unsupported CICP matrix_coefficients {matrix_code}; expected 0 (Identity RGB)"
        ));
    }
    if full_range_code != 1 {
        return Err(format!(
            "unsupported CICP full_range flag {full_range_code}; expected 1 (full-range SDR)"
        ));
    }

    let color_primaries = match primaries_code {
        1 => CicpColorPrimaries::Bt709,
        4 => CicpColorPrimaries::Bt470M,
        5 => CicpColorPrimaries::Bt470Bg,
        6 => CicpColorPrimaries::Bt601,
        7 => CicpColorPrimaries::Smpte240,
        9 => CicpColorPrimaries::Bt2020,
        11 => CicpColorPrimaries::Smpte431,
        12 => CicpColorPrimaries::Smpte432,
        22 => CicpColorPrimaries::Ebu3213,
        other => {
            return Err(format!(
                "unsupported or unspecified CICP color_primaries {other}"
            ));
        }
    };

    let transfer_characteristics = match transfer_code {
        1 => TransferCharacteristics::Bt709,
        4 => TransferCharacteristics::Bt470M,
        5 => TransferCharacteristics::Bt470Bg,
        6 => TransferCharacteristics::Bt601,
        7 => TransferCharacteristics::Smpte240,
        8 => TransferCharacteristics::Linear,
        13 => TransferCharacteristics::Srgb,
        16 => {
            return Err(
                "HDR PQ CICP transfer_characteristics (16) is outside the SDR sRGB contract"
                    .to_string(),
            );
        }
        18 => {
            return Err(
                "HDR HLG CICP transfer_characteristics (18) is outside the SDR sRGB contract"
                    .to_string(),
            );
        }
        other => {
            return Err(format!(
                "unsupported or unspecified CICP transfer_characteristics {other}"
            ));
        }
    };

    if color_primaries == CicpColorPrimaries::Bt709
        && transfer_characteristics == TransferCharacteristics::Srgb
    {
        return Ok(false);
    }
    if pixels.is_empty() {
        return Ok(false);
    }

    let source_profile = ColorProfile::new_from_cicp(CicpProfile {
        color_primaries,
        transfer_characteristics,
        matrix_coefficients: MatrixCoefficients::Identity,
        full_range: true,
    });
    if source_profile.red_trc.is_none()
        || source_profile.green_trc.is_none()
        || source_profile.blue_trc.is_none()
    {
        return Err(format!(
            "failed to synthesize SDR RGB profile from CICP {cicp:?}"
        ));
    }

    apply_rgb_profile_transform_to_srgb(pixels, &source_profile)?;
    Ok(true)
}

pub fn srgb_preview_encoder(preset: Preset) -> Encoder {
    Encoder::new(preset).icc_profile(srgb_v4_profile().to_vec())
}

fn png_encodable_srgb_image(image: &DynamicImage) -> std::borrow::Cow<'_, DynamicImage> {
    match image {
        DynamicImage::ImageRgb8(_)
        | DynamicImage::ImageRgba8(_)
        | DynamicImage::ImageRgb16(_)
        | DynamicImage::ImageRgba16(_) => std::borrow::Cow::Borrowed(image),
        _ if image.color().has_alpha() => {
            std::borrow::Cow::Owned(DynamicImage::ImageRgba8(image.to_rgba8()))
        }
        _ => std::borrow::Cow::Owned(DynamicImage::ImageRgb8(image.to_rgb8())),
    }
}

/// Encodes a display-encoded sRGB image into PNG bytes with the bundled sRGB v4
/// ICC profile embedded (for WebView previews and lossless tagged PNG output).
pub fn encode_srgb_png_bytes(image: &DynamicImage) -> Result<Vec<u8>, String> {
    let encodable = png_encodable_srgb_image(image);
    let mut bytes = Vec::new();
    let mut encoder = PngEncoder::new(Cursor::new(&mut bytes));
    encoder
        .set_icc_profile(srgb_v4_profile().to_vec())
        .map_err(|error| format!("Failed to attach sRGB ICC profile to PNG: {error}"))?;
    encodable
        .write_with_encoder(encoder)
        .map_err(|error| format!("Failed to encode sRGB PNG: {error}"))?;
    Ok(bytes)
}

/// Encodes a display-encoded sRGB image into JPEG bytes with the bundled sRGB
/// v4 ICC profile embedded (for WebView previews that use `image::JpegEncoder`).
pub fn encode_srgb_jpeg_bytes(image: &DynamicImage, quality: u8) -> Result<Vec<u8>, String> {
    let rgb = image.to_rgb8();
    let mut bytes = Vec::new();
    let mut encoder = JpegEncoder::new_with_quality(&mut bytes, quality.clamp(1, 100));
    encoder
        .set_icc_profile(srgb_v4_profile().to_vec())
        .map_err(|error| format!("Failed to attach sRGB ICC profile to JPEG: {error}"))?;
    rgb.write_with_encoder(encoder)
        .map_err(|error| format!("Failed to encode sRGB JPEG: {error}"))?;
    Ok(bytes)
}

/// Saves a display-encoded sRGB image to disk with the bundled sRGB v4 ICC
/// profile embedded for PNG, JPEG, and TIFF outputs.
///
/// Linear RAW buffers must NOT call this helper.
pub fn save_srgb_tagged_image(image: &DynamicImage, path: &Path) -> Result<(), String> {
    let format = ImageFormat::from_path(path).unwrap_or(ImageFormat::Png);
    match format {
        ImageFormat::Png => {
            let file = File::create(path)
                .map_err(|error| format!("Failed to create '{}': {error}", path.display()))?;
            let encodable = png_encodable_srgb_image(image);
            let mut encoder = PngEncoder::new(BufWriter::new(file));
            encoder
                .set_icc_profile(srgb_v4_profile().to_vec())
                .map_err(|error| format!("Failed to attach sRGB ICC profile to PNG: {error}"))?;
            encodable
                .write_with_encoder(encoder)
                .map_err(|error| format!("Failed to save tagged PNG '{}': {error}", path.display()))
        }
        ImageFormat::Jpeg => {
            let file = File::create(path)
                .map_err(|error| format!("Failed to create '{}': {error}", path.display()))?;
            let mut encoder = JpegEncoder::new_with_quality(BufWriter::new(file), 94);
            encoder
                .set_icc_profile(srgb_v4_profile().to_vec())
                .map_err(|error| format!("Failed to attach sRGB ICC profile to JPEG: {error}"))?;
            image
                .to_rgb8()
                .write_with_encoder(encoder)
                .map_err(|error| {
                    format!("Failed to save tagged JPEG '{}': {error}", path.display())
                })
        }
        ImageFormat::Tiff => {
            let file = File::create(path)
                .map_err(|error| format!("Failed to create '{}': {error}", path.display()))?;
            let mut encoder = TiffEncoder::new(BufWriter::new(file));
            encoder
                .set_icc_profile(srgb_v4_profile().to_vec())
                .map_err(|error| format!("Failed to attach sRGB ICC profile to TIFF: {error}"))?;
            image.write_with_encoder(encoder).map_err(|error| {
                format!("Failed to save tagged TIFF '{}': {error}", path.display())
            })
        }
        _ => image
            .save_with_format(path, format)
            .map_err(|error| format!("Failed to save image '{}': {error}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use image::{ImageDecoder, ImageFormat, ImageReader};
    use moxcms::ColorProfile;
    use mozjpeg_rs::Preset;
    use sha2::{Digest, Sha256};

    use super::*;

    fn assert_close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    fn synthetic_icc_header(color_space: &[u8; 4], declared_size: usize) -> Vec<u8> {
        let mut profile = vec![0u8; declared_size.max(128)];
        profile[0..4].copy_from_slice(&(declared_size as u32).to_be_bytes());
        profile[16..20].copy_from_slice(color_space);
        profile[36..40].copy_from_slice(b"acsp");
        profile
    }

    #[test]
    fn bundled_srgb_profile_has_stable_identity_and_valid_header() {
        let profile = srgb_v4_profile();
        validate_icc_profile(profile).expect("valid bundled profile");
        assert_eq!(profile.len(), 480);
        assert_eq!(
            hex::encode(Sha256::digest(profile)),
            "c56e1685d888f5edb92fe07f2750f387f8fe8e91b32ff8fb0b56bfbbb9458353"
        );
    }

    #[test]
    fn srgb_transfer_matches_reference_values() {
        assert_close(srgb_to_linear_channel(0.0), 0.0, 1.0e-7);
        assert_close(srgb_to_linear_channel(0.04045), 0.003_130_805, 1.0e-7);
        assert_close(srgb_to_linear_channel(0.5), 0.214_041_14, 1.0e-7);
        assert_close(srgb_to_linear_channel(1.0), 1.0, 1.0e-7);

        assert_close(linear_to_srgb_channel(0.0), 0.0, 1.0e-7);
        assert_close(linear_to_srgb_channel(0.003_130_8), 0.040_449_936, 1.0e-7);
        assert_close(linear_to_srgb_channel(0.214_041_14), 0.5, 1.0e-7);
        assert_close(linear_to_srgb_channel(1.0), 1.0, 1.0e-7);
    }

    #[test]
    fn srgb_transfer_round_trips_display_range() {
        for step in 0..=4096 {
            let encoded = step as f32 / 4096.0;
            let round_trip = linear_to_srgb_channel(srgb_to_linear_channel(encoded));
            assert_close(round_trip, encoded, 2.0e-6);
        }
    }

    #[test]
    fn display_p3_pixels_are_normalized_to_encoded_srgb() {
        let profile = ColorProfile::new_display_p3()
            .encode()
            .expect("encode deterministic Display P3 profile");
        let mut pixels = [0.5, 0.25, 0.75];

        assert!(normalize_encoded_rgb_profile_to_srgb(&mut pixels, &profile).unwrap());
        assert_close(pixels[0], 0.537, 0.003);
        assert_close(pixels[1], 0.232, 0.003);
        assert_close(pixels[2], 0.777, 0.003);
    }

    #[test]
    fn malformed_or_oversized_profiles_do_not_mutate_pixels() {
        let original = [0.2, 0.4, 0.6];

        let mut malformed_pixels = original;
        assert!(
            normalize_encoded_rgb_profile_to_srgb(&mut malformed_pixels, b"not an ICC profile")
                .is_err()
        );
        assert_eq!(malformed_pixels, original);

        let mut oversized_pixels = original;
        let oversized_profile = vec![0; MAX_EMBEDDED_ICC_BYTES + 1];
        assert!(
            normalize_encoded_rgb_profile_to_srgb(&mut oversized_pixels, &oversized_profile)
                .is_err()
        );
        assert_eq!(oversized_pixels, original);

        // Valid 128-byte RGB header with corrupt tag table must also fail closed without mutating pixels.
        let mut corrupt_rgb = synthetic_icc_header(b"RGB ", 160);
        corrupt_rgb[128..132].copy_from_slice(&999u32.to_be_bytes());
        let mut corrupt_pixels = original;
        assert!(normalize_encoded_rgb_profile_to_srgb(&mut corrupt_pixels, &corrupt_rgb).is_err());
        assert_eq!(corrupt_pixels, original);
    }

    #[test]
    fn cmyk_profile_is_classified_and_rejected_without_mutating_pixels() {
        let cmyk_header = synthetic_icc_header(b"CMYK", 128);
        assert_eq!(
            inspect_icc_profile_header(&cmyk_header).unwrap(),
            IccColorSpace::Cmyk
        );

        let original = [0.25, 0.5, 0.75];
        let mut rgb_pixels = original;
        let err = normalize_encoded_rgb_profile_to_srgb(&mut rgb_pixels, &cmyk_header)
            .expect_err("CMYK profile must not be applied to RGB pixels");
        assert!(err.contains("CMYK"));
        assert_eq!(rgb_pixels, original);

        let mut gray_pixels = [0.5, 0.5, 0.5];
        let gray_err = normalize_encoded_gray_profile_to_srgb(&mut gray_pixels, &cmyk_header)
            .expect_err("CMYK profile must not be applied to Gray pixels");
        assert!(gray_err.contains("CMYK"));
        assert_eq!(gray_pixels, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn gray_profile_normalizes_grayscale_pixels_and_rejects_mismatches() {
        let gray_profile = ColorProfile::new_gray_with_gamma(1.8)
            .encode()
            .expect("encode deterministic Gray Gamma 1.8 profile");
        assert_eq!(
            inspect_icc_profile_header(&gray_profile).unwrap(),
            IccColorSpace::Gray
        );

        // 0.5^1.8 = 0.28717459 linear -> sRGB OETF ≈ 0.5712
        let mut pixels = [0.5_f32, 0.5, 0.5];
        assert!(normalize_encoded_gray_profile_to_srgb(&mut pixels, &gray_profile).unwrap());
        let expected = linear_to_srgb_channel(0.5_f32.powf(1.8));
        assert_close(pixels[0], expected, 0.005);
        assert_close(pixels[1], expected, 0.005);
        assert_close(pixels[2], expected, 0.005);

        // Applying an RGB profile through the Gray helper fails closed without mutating pixels.
        let original = [0.3, 0.3, 0.3];
        let mut mismatch = original;
        assert!(normalize_encoded_gray_profile_to_srgb(&mut mismatch, srgb_v4_profile()).is_err());
        assert_eq!(mismatch, original);
    }

    #[test]
    fn cicp_sdr_profiles_normalize_and_hdr_or_invalid_cicp_fail_closed() {
        // Standard sRGB CICP is an explicit no-op.
        let mut srgb_pixels = [0.5, 0.25, 0.75];
        assert!(!normalize_encoded_rgb_cicp_to_srgb(&mut srgb_pixels, [1, 13, 0, 1]).unwrap());
        assert_eq!(srgb_pixels, [0.5, 0.25, 0.75]);

        // Display P3 CICP ([12, 13, 0, 1]) matches Display P3 ICC conversion.
        let mut p3_pixels = [0.5, 0.25, 0.75];
        assert!(normalize_encoded_rgb_cicp_to_srgb(&mut p3_pixels, [12, 13, 0, 1]).unwrap());
        assert_close(p3_pixels[0], 0.537, 0.004);
        assert_close(p3_pixels[1], 0.232, 0.004);
        assert_close(p3_pixels[2], 0.777, 0.004);

        // Linear BT.709/sRGB CICP ([1, 8, 0, 1]) encodes linear 0.21404114 -> ~0.5 sRGB.
        let mut linear_pixels = [0.214_041_14, 0.214_041_14, 0.214_041_14];
        assert!(normalize_encoded_rgb_cicp_to_srgb(&mut linear_pixels, [1, 8, 0, 1]).unwrap());
        assert_close(linear_pixels[0], 0.5, 0.003);

        // HDR PQ/HLG, narrow-range, non-RGB matrix, and unspecified/reserved tuples must fail closed.
        let original = [0.4, 0.5, 0.6];
        for unsupported in [
            [9, 16, 0, 1],  // BT.2020 PQ (HDR)
            [9, 18, 0, 1],  // BT.2020 HLG (HDR)
            [12, 13, 1, 1], // YCbCr matrix
            [12, 13, 0, 0], // narrow range
            [2, 13, 0, 1],  // unspecified primaries
            [12, 2, 0, 1],  // unspecified transfer
            [0, 13, 0, 1],  // reserved primaries
        ] {
            let mut pixels = original;
            assert!(
                normalize_encoded_rgb_cicp_to_srgb(&mut pixels, unsupported).is_err(),
                "CICP {unsupported:?} must fail closed"
            );
            assert_eq!(pixels, original);
        }
    }

    #[test]
    fn webview_preview_encoder_embeds_the_srgb_profile() {
        let jpeg = srgb_preview_encoder(Preset::BaselineFastest)
            .quality(90)
            .encode_rgb(&[128, 64, 32], 1, 1)
            .expect("encode tagged preview");
        let mut decoder = ImageReader::with_format(Cursor::new(jpeg), ImageFormat::Jpeg)
            .into_decoder()
            .expect("decode tagged preview");

        assert_eq!(
            decoder.icc_profile().expect("read preview ICC").as_deref(),
            Some(srgb_v4_profile())
        );
    }

    #[test]
    fn tagged_preview_and_save_helpers_embed_srgb_icc_and_preserve_pixels() {
        let rgb = image::RgbImage::from_fn(16, 16, |x, y| {
            image::Rgb([
                if x < 8 { 48 } else { 200 },
                if y < 8 { 96 } else { 164 },
                128,
            ])
        });
        let dynamic = DynamicImage::ImageRgb8(rgb.clone());

        let png_bytes = encode_srgb_png_bytes(&dynamic).expect("encode tagged PNG preview");
        let mut png_decoder = ImageReader::with_format(Cursor::new(&png_bytes), ImageFormat::Png)
            .into_decoder()
            .expect("decode tagged PNG preview");
        assert_eq!(
            png_decoder.icc_profile().expect("read PNG ICC").as_deref(),
            Some(srgb_v4_profile())
        );
        let decoded_png = image::load_from_memory_with_format(&png_bytes, ImageFormat::Png)
            .expect("decode PNG pixels")
            .to_rgb8();
        assert_eq!(decoded_png, rgb);

        let jpeg_bytes = encode_srgb_jpeg_bytes(&dynamic, 95).expect("encode tagged JPEG preview");
        let mut jpeg_decoder =
            ImageReader::with_format(Cursor::new(&jpeg_bytes), ImageFormat::Jpeg)
                .into_decoder()
                .expect("decode tagged JPEG preview");
        assert_eq!(
            jpeg_decoder
                .icc_profile()
                .expect("read JPEG ICC")
                .as_deref(),
            Some(srgb_v4_profile())
        );

        let dir = tempfile::tempdir().expect("tempdir");
        for (file_name, format) in [
            ("tagged.png", ImageFormat::Png),
            ("tagged.jpg", ImageFormat::Jpeg),
            ("tagged.tiff", ImageFormat::Tiff),
        ] {
            let path = dir.path().join(file_name);
            save_srgb_tagged_image(&dynamic, &path).expect("save tagged image");
            let saved_bytes = std::fs::read(&path).expect("read saved image");
            let saved_icc = match format {
                ImageFormat::Tiff => {
                    image::codecs::tiff::TiffDecoder::new(Cursor::new(&saved_bytes))
                        .expect("decode saved TIFF")
                        .icc_profile()
                        .expect("read saved TIFF ICC")
                }
                _ => ImageReader::with_format(Cursor::new(&saved_bytes), format)
                    .into_decoder()
                    .expect("decode saved image")
                    .icc_profile()
                    .expect("read saved ICC"),
            };
            assert_eq!(
                saved_icc.as_deref(),
                Some(srgb_v4_profile()),
                "saved {file_name} must embed sRGB v4 ICC"
            );
            if matches!(format, ImageFormat::Png | ImageFormat::Tiff) {
                let decoded = image::load_from_memory_with_format(&saved_bytes, format)
                    .expect("decode lossless saved image")
                    .to_rgb8();
                assert_eq!(decoded, rgb, "lossless {file_name} must preserve exact RGB");
            }
        }
    }

    #[test]
    fn linear_node_reference_chart_preserves_display_p3_and_exposure_linearity() {
        // Synthetic linear-node reference patches covering black, dark toe,
        // 18% middle gray, highlight, and saturated primaries.
        let linear_patches: [[f32; 3]; 6] = [
            [0.0, 0.0, 0.0],
            [0.002, 0.002, 0.002],
            [0.18, 0.18, 0.18],
            [0.45, 0.45, 0.45],
            [0.35, 0.08, 0.04],
            [0.06, 0.28, 0.42],
        ];
        for linear in linear_patches {
            let encoded = linear.map(linear_to_srgb_channel);
            let recovered_linear = encoded.map(srgb_to_linear_channel);
            for c in 0..3 {
                assert_close(recovered_linear[c], linear[c], 1.0e-6);
            }

            // +1 EV followed by -1 EV in linear space must be exact without tone shift.
            let boosted = recovered_linear.map(|v| v * 2.0);
            let restored = boosted.map(|v| v * 0.5);
            for c in 0..3 {
                assert_close(restored[c], linear[c], 1.0e-6);
                let u8_q = (linear_to_srgb_channel(restored[c]).clamp(0.0, 1.0) * 255.0).round();
                let u8_ref = (encoded[c].clamp(0.0, 1.0) * 255.0).round();
                assert_eq!(u8_q, u8_ref);
            }
        }

        // Display P3 neutral ramp must stay neutral after ICC -> sRGB -> linear EOTF.
        let p3_profile = ColorProfile::new_display_p3()
            .encode()
            .expect("encode Display P3 profile");
        for gray_linear in [0.002_f32, 0.18, 0.5] {
            let encoded = linear_to_srgb_channel(gray_linear);
            let mut pixel = [encoded, encoded, encoded];
            normalize_encoded_rgb_profile_to_srgb(&mut pixel, &p3_profile)
                .expect("normalize neutral P3 patch");
            let out_linear = pixel.map(srgb_to_linear_channel);
            assert_close(out_linear[0], gray_linear, 0.004);
            assert_close(out_linear[1], gray_linear, 0.004);
            assert_close(out_linear[2], gray_linear, 0.004);
            assert!((out_linear[0] - out_linear[1]).abs() <= 0.002);
            assert!((out_linear[1] - out_linear[2]).abs() <= 0.002);
        }
    }
}
