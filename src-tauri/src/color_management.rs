use std::sync::OnceLock;

use base64::{Engine as _, engine::general_purpose};

/// Compact ICC Profiles' CC0 sRGB v4 profile.
/// Source: https://github.com/saucecontrol/Compact-ICC-Profiles/blob/master/profiles/sRGB-v4.icc
const SRGB_V4_BASE64: &str = "AAAB4GxjbXMEIAAAbW50clJHQiBYWVogB+IAAwAUAAkADgAdYWNzcE1TRlQAAAAAc2F3c2N0cmwAAAAAAAAAAAAAAAAAAPbWAAEAAAAA0y1oYW5keem/Vlo+AbaDI4VVRvdPqgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKZGVzYwAAAPwAAAAkY3BydAAAASAAAAAid3RwdAAAAUQAAAAUY2hhZAAAAVgAAAAsclhZWgAAAYQAAAAUZ1hZWgAAAZgAAAAUYlhZWgAAAawAAAAUclRSQwAAAcAAAAAgZ1RSQwAAAcAAAAAgYlRSQwAAAcAAAAAgbWx1YwAAAAAAAAABAAAADGVuVVMAAAAIAAAAHABzAFIARwBCbWx1YwAAAAAAAAABAAAADGVuVVMAAAAGAAAAHABDAEMAMAAAWFlaIAAAAAAAAPbWAAEAAAAA0y1zZjMyAAAAAAABDD8AAAXd///zJgAAB5AAAP2S///7of///aIAAAPcAADAcVhZWiAAAAAAAABvoAAAOPIAAAOPWFlaIAAAAAAAAGKWAAC3iQAAGNpYWVogAAAAAAAAJKAAAA+FAAC2xHBhcmEAAAAAAAMAAAACZmkAAPKnAAANWQAAE9AAAApb";

static SRGB_V4_PROFILE: OnceLock<Vec<u8>> = OnceLock::new();

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
}
