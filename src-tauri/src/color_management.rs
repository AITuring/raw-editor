use std::sync::OnceLock;

use base64::{Engine as _, engine::general_purpose};
use moxcms::{ColorProfile, DataColorSpace, Layout, ParsingOptions, TransformOptions};
use mozjpeg_rs::{Encoder, Preset};

/// Compact ICC Profiles' CC0 sRGB v4 profile.
/// Source: https://github.com/saucecontrol/Compact-ICC-Profiles/blob/master/profiles/sRGB-v4.icc
const SRGB_V4_BASE64: &str = "AAAB4GxjbXMEIAAAbW50clJHQiBYWVogB+IAAwAUAAkADgAdYWNzcE1TRlQAAAAAc2F3c2N0cmwAAAAAAAAAAAAAAAAAAPbWAAEAAAAA0y1oYW5keem/Vlo+AbaDI4VVRvdPqgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKZGVzYwAAAPwAAAAkY3BydAAAASAAAAAid3RwdAAAAUQAAAAUY2hhZAAAAVgAAAAsclhZWgAAAYQAAAAUZ1hZWgAAAZgAAAAUYlhZWgAAAawAAAAUclRSQwAAAcAAAAAgZ1RSQwAAAcAAAAAgYlRSQwAAAcAAAAAgbWx1YwAAAAAAAAABAAAADGVuVVMAAAAIAAAAHABzAFIARwBCbWx1YwAAAAAAAAABAAAADGVuVVMAAAAGAAAAHABDAEMAMAAAWFlaIAAAAAAAAPbWAAEAAAAA0y1zZjMyAAAAAAABDD8AAAXd///zJgAAB5AAAP2S///7of///aIAAAPcAADAcVhZWiAAAAAAAABvoAAAOPIAAAOPWFlaIAAAAAAAAGKWAAC3iQAAGNpYWVogAAAAAAAAJKAAAA+FAAC2xHBhcmEAAAAAAAMAAAACZmkAAPKnAAANWQAAE9AAAApb";

static SRGB_V4_PROFILE: OnceLock<Vec<u8>> = OnceLock::new();

pub const MAX_EMBEDDED_ICC_BYTES: usize = 4 * 1024 * 1024;

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

pub fn validate_icc_profile(profile: &[u8]) -> Result<(), String> {
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
    if &profile[16..20] != b"RGB " {
        return Err("ICC profile is not an RGB profile".to_string());
    }
    Ok(())
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

    let parsing_options = ParsingOptions {
        max_profile_size: MAX_EMBEDDED_ICC_BYTES + 1,
        max_allowed_clut_size: MAX_EMBEDDED_ICC_BYTES,
        max_allowed_trc_size: 65_536,
    };
    let source_profile = ColorProfile::new_from_slice_with_options(profile, parsing_options)
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

pub fn srgb_preview_encoder(preset: Preset) -> Encoder {
    Encoder::new(preset).icc_profile(srgb_v4_profile().to_vec())
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
}
