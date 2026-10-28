//! EXIF value conversion and display formatting.
//!
//! Rational conversions for little_exif, sidecar GPS parsing, creation-date
//! parsing and the shutter/lens strings shown in the UI.

use chrono::DateTime;
use chrono::NaiveDateTime;
use chrono::Utc;
use little_exif::rational::iR64;
use little_exif::rational::uR64;

pub(super) fn to_ur64(val: &exif::Rational) -> uR64 {
    uR64 {
        nominator: val.num,
        denominator: val.denom,
    }
}

pub(super) fn to_ir64(val: &exif::SRational) -> iR64 {
    iR64 {
        nominator: val.num,
        denominator: val.denom,
    }
}

pub(super) fn decimal_to_ur64(value: f64) -> Option<uR64> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    const DENOMINATOR: u32 = 1_000_000;
    let nominator = (value * f64::from(DENOMINATOR)).round();
    if nominator > f64::from(u32::MAX) {
        return None;
    }
    Some(uR64 {
        nominator: nominator as u32,
        denominator: DENOMINATOR,
    })
}

pub(super) fn parse_sidecar_gps_coordinate(value: &str) -> Option<Vec<uR64>> {
    let values = value
        .split(|character: char| !(character.is_ascii_digit() || character == '.'))
        .filter(|part| !part.is_empty())
        .filter_map(|part| part.parse::<f64>().ok())
        .take(3)
        .map(decimal_to_ur64)
        .collect::<Option<Vec<_>>>()?;
    (values.len() == 3).then_some(values)
}

fn clean_creation_datetime_str(s: &str) -> &str {
    s.trim().trim_matches('"').trim_matches('\'').trim()
}

pub(super) fn fmt_date_str(s: String) -> String {
    if let Some(dt) = parse_creation_datetime(&s) {
        return dt.format("%Y-%m-%d %H:%M:%S").to_string();
    }
    clean_creation_datetime_str(&s).to_string()
}

fn normalize_creation_datetime(s: &str) -> Option<String> {
    let normalized = s.replace('T', " ");
    let (date, time) = normalized.split_once(' ')?;
    Some(format!("{} {}", date.replace(':', "-"), time))
}

pub(super) fn parse_creation_datetime(s: &str) -> Option<NaiveDateTime> {
    let clean = clean_creation_datetime_str(s);
    if clean.is_empty() {
        return None;
    }

    let normalized = normalize_creation_datetime(clean);
    for candidate in std::iter::once(clean).chain(normalized.as_deref()) {
        for format in [
            "%Y:%m:%d %H:%M:%S",
            "%Y:%m:%d %H:%M:%S%.f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S%.f",
        ] {
            if let Ok(dt) = NaiveDateTime::parse_from_str(candidate, format) {
                return Some(dt);
            }
        }
    }

    None
}

pub(super) fn parse_creation_field(field: &exif::Field) -> Option<DateTime<Utc>> {
    parse_creation_datetime(&field.display_value().to_string())
        .map(|dt| DateTime::from_naive_utc_and_offset(dt, Utc))
}

pub(super) fn parse_raw_creation_date(date_str: Option<&str>) -> Option<DateTime<Utc>> {
    parse_creation_datetime(date_str?).map(|dt| DateTime::from_naive_utc_and_offset(dt, Utc))
}

pub(super) fn clean_ascii_value(value: &exif::Value) -> Option<String> {
    let exif::Value::Ascii(ref components) = *value else {
        return None;
    };

    let cleaned: Vec<String> = components
        .iter()
        .map(|c| {
            String::from_utf8_lossy(c)
                .trim_matches(char::from(0))
                .trim()
                .to_string()
        })
        .filter(|s| !s.is_empty())
        .collect();

    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned.join(" "))
    }
}

pub(super) fn rational_to_f32_checked(r: &exif::Rational) -> Option<f32> {
    if r.denom == 0 {
        None
    } else {
        Some(r.num as f32 / r.denom as f32)
    }
}

pub(super) fn rawler_rational_to_f32_checked(r: &rawler::formats::tiff::Rational) -> Option<f32> {
    if r.d == 0 {
        None
    } else {
        Some(r.n as f32 / r.d as f32)
    }
}

pub(super) fn format_min_max(min: f32, max: f32, tolerance: f32) -> String {
    if (min - max).abs() < tolerance {
        format!("{min}")
    } else {
        format!("{min}-{max}")
    }
}

/// Keep shutter values readable when an EXIF library emits a reciprocal
/// fraction with a floating-point denominator (for example, `1/1.6666667 s`
/// for a 0.6 second exposure). The stored rational remains untouched; this is
/// only the presentation string used by the UI and sidecar metadata display.
pub(super) fn format_exposure_time_seconds(seconds: f64) -> Option<String> {
    if !seconds.is_finite() || seconds <= 0.0 {
        return None;
    }

    let compact = |value: f64| {
        let rounded = (value * 100.0).round() / 100.0;
        let mut formatted = format!("{rounded:.2}");
        while formatted.contains('.') && formatted.ends_with('0') {
            formatted.pop();
        }
        if formatted.ends_with('.') {
            formatted.pop();
        }
        formatted
    };

    if seconds >= 1.0 {
        return Some(format!("{} s", compact(seconds)));
    }

    let denominator = 1.0 / seconds;
    if denominator >= 2.0 {
        Some(format!("1/{} s", compact(denominator)))
    } else {
        Some(format!("{} s", compact(seconds)))
    }
}

pub(super) fn format_exif_exposure_time(numerator: u32, denominator: u32) -> Option<String> {
    if denominator == 0 {
        return None;
    }
    format_exposure_time_seconds(numerator as f64 / denominator as f64)
}

pub(super) fn format_lens_specification(components: &[exif::Rational]) -> Option<String> {
    if components.len() < 4 {
        return None;
    }

    let focal_min = rational_to_f32_checked(&components[0]);
    let focal_max = rational_to_f32_checked(&components[1]);
    let (focal_min, focal_max) = match (focal_min, focal_max) {
        (Some(min), Some(max)) => (min, max),
        _ => return None,
    };

    let mut spec = format!("{} mm", format_min_max(focal_min, focal_max, 0.01));

    let aperture_min = rational_to_f32_checked(&components[2]);
    let aperture_max = rational_to_f32_checked(&components[3]);
    if let (Some(amin), Some(amax)) = (aperture_min, aperture_max) {
        spec.push_str(&format!(", f/{}", format_min_max(amin, amax, 0.01)));
    }

    Some(spec)
}
