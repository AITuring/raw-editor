use std::sync::OnceLock;

use base64::{Engine as _, engine::general_purpose};

/// Compact ICC Profiles' CC0 sRGB v4 profile.
/// Source: https://github.com/saucecontrol/Compact-ICC-Profiles/blob/master/profiles/sRGB-v4.icc
const SRGB_V4_BASE64: &str = "AAAB4GxjbXMEIAAAbW50clJHQiBYWVogB+IAAwAUAAkADgAdYWNzcE1TRlQAAAAAc2F3c2N0cmwAAAAAAAAAAAAAAAAAAPbWAAEAAAAA0y1oYW5keem/Vlo+AbaDI4VVRvdPqgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKZGVzYwAAAPwAAAAkY3BydAAAASAAAAAid3RwdAAAAUQAAAAUY2hhZAAAAVgAAAAsclhZWgAAAYQAAAAUZ1hZWgAAAZgAAAAUYlhZWgAAAawAAAAUclRSQwAAAcAAAAAgZ1RSQwAAAcAAAAAgYlRSQwAAAcAAAAAgbWx1YwAAAAAAAAABAAAADGVuVVMAAAAIAAAAHABzAFIARwBCbWx1YwAAAAAAAAABAAAADGVuVVMAAAAGAAAAHABDAEMAMAAAWFlaIAAAAAAAAPbWAAEAAAAA0y1zZjMyAAAAAAABDD8AAAXd///zJgAAB5AAAP2S///7of///aIAAAPcAADAcVhZWiAAAAAAAABvoAAAOPIAAAOPWFlaIAAAAAAAAGKWAAC3iQAAGNpYWVogAAAAAAAAJKAAAA+FAAC2xHBhcmEAAAAAAAMAAAACZmkAAPKnAAANWQAAE9AAAApb";

static SRGB_V4_PROFILE: OnceLock<Vec<u8>> = OnceLock::new();

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

#[cfg(test)]
mod tests {
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
}
