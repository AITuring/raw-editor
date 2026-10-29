//! Metadata written into exported files.
//!
//! Builds the retained, stripped or overridden EXIF set for an export and
//! writes it into encoded bytes or the streaming TIFF writer.

use crate::formats::is_raw_file;
use little_exif::exif_tag::ExifTag;
use little_exif::filetype::FileExtension;
use little_exif::metadata::Metadata;
use little_exif::rational::iR64;
use little_exif::rational::uR64;
use std::path::Path;

use super::{decimal_to_ur64, parse_sidecar_gps_coordinate, read_rrexif_sidecar, to_ir64, to_ur64};

#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportMetadataOverrides {
    pub artist: Option<String>,
    pub contact: Option<String>,
    pub copyright: Option<String>,
    pub description: Option<String>,
}

impl ExportMetadataOverrides {
    pub(super) fn normalized(value: &Option<String>, max_chars: usize) -> Option<String> {
        let value = value.as_deref()?.trim();
        if value.is_empty() {
            return None;
        }
        Some(value.chars().take(max_chars).collect())
    }

    pub fn has_instructions(&self) -> bool {
        self.artist.is_some()
            || self.contact.is_some()
            || self.copyright.is_some()
            || self.description.is_some()
    }
}

fn build_export_metadata(
    original_path_str: &str,
    output_format: &str,
    keep_metadata: bool,
    strip_gps: bool,
    overrides: Option<&ExportMetadataOverrides>,
) -> Result<Option<(Metadata, FileExtension)>, String> {
    let has_overrides = overrides.is_some_and(ExportMetadataOverrides::has_instructions);
    if !keep_metadata && !has_overrides {
        return Ok(None);
    }

    let original_path = Path::new(original_path_str);
    if keep_metadata && !original_path.exists() && !has_overrides {
        return Ok(None);
    }

    let file_type = match output_format.to_lowercase().as_str() {
        "jpg" | "jpeg" => FileExtension::JPEG,
        "png" => FileExtension::PNG {
            as_zTXt_chunk: true,
        },
        "tif" | "tiff" => FileExtension::TIFF,
        _ => return Ok(None),
    };

    let mut metadata = Metadata::new();
    let mut source_read_success = false;

    if keep_metadata && let Some(map) = read_rrexif_sidecar(original_path) {
        source_read_success = true;

        let clean_s = |s: &String| s.replace('"', "").trim().to_string();

        let parse_ur64 = |s: &str| -> Option<uR64> {
            let cleaned_string = s
                .replace("f/", "")
                .replace(" s", "")
                .replace(" mm", "")
                .replace("\"", "");

            let val = cleaned_string.trim();

            if val.contains('/') {
                let parts: Vec<&str> = val.split('/').collect();
                if parts.len() == 2
                    && let (Ok(n), Ok(d)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>())
                {
                    return Some(uR64 {
                        nominator: n,
                        denominator: d,
                    });
                }
            } else if let Ok(f) = val.parse::<f32>() {
                return Some(uR64 {
                    nominator: (f * 1000.0) as u32,
                    denominator: 1000,
                });
            }
            None
        };
        if let Some(val) = map.get("Make") {
            metadata.set_tag(ExifTag::Make(clean_s(val)));
        }
        if let Some(val) = map.get("Model") {
            metadata.set_tag(ExifTag::Model(clean_s(val)));
        }
        if let Some(val) = map.get("LensMake") {
            metadata.set_tag(ExifTag::LensMake(clean_s(val)));
        }
        if let Some(val) = map.get("LensModel") {
            metadata.set_tag(ExifTag::LensModel(clean_s(val)));
        }
        if let Some(val) = map.get("Artist") {
            metadata.set_tag(ExifTag::Artist(clean_s(val)));
        }
        if let Some(val) = map.get("Copyright") {
            metadata.set_tag(ExifTag::Copyright(clean_s(val)));
        }
        if let Some(val) = map.get("UserComment") {
            metadata.set_tag(ExifTag::UserComment(clean_s(val).into_bytes()));
        }
        if let Some(val) = map.get("ImageDescription") {
            metadata.set_tag(ExifTag::ImageDescription(clean_s(val)));
        }
        if let Some(val) = map.get("DateTimeOriginal") {
            metadata.set_tag(ExifTag::DateTimeOriginal(clean_s(val)));
        }
        if let Some(val) = map.get("CreateDate") {
            metadata.set_tag(ExifTag::CreateDate(clean_s(val)));
        }
        if let Some(val) = map.get("FNumber")
            && let Some(ur) = parse_ur64(val)
        {
            metadata.set_tag(ExifTag::FNumber(vec![ur]));
        }
        if let Some(val) = map.get("ExposureTime")
            && let Some(ur) = parse_ur64(val)
        {
            metadata.set_tag(ExifTag::ExposureTime(vec![ur]));
        }
        if let Some(val) = map.get("FocalLength")
            && let Some(ur) = parse_ur64(val)
        {
            metadata.set_tag(ExifTag::FocalLength(vec![ur]));
        }
        if let Some(val) = map.get("FocalLengthIn35mmFilm") {
            let cleaned = val.replace(" mm", "").replace("\"", "");
            let trimmed = cleaned.trim();
            if let Ok(f_val) = trimmed.parse::<f32>() {
                metadata.set_tag(ExifTag::FocalLengthIn35mmFormat(vec![f_val.round() as u16]));
            }
        }
        if let Some(val) = map.get("ISOSpeed").or(map.get("PhotographicSensitivity"))
            && let Ok(iso) = val.replace('"', "").trim().parse::<u16>()
        {
            metadata.set_tag(ExifTag::ISO(vec![iso]));
        }
        if !strip_gps {
            if let Some(value) = map
                .get("GPSLatitude")
                .and_then(|value| parse_sidecar_gps_coordinate(value))
            {
                metadata.set_tag(ExifTag::GPSLatitude(value));
            }
            if let Some(value) = map.get("GPSLatitudeRef") {
                metadata.set_tag(ExifTag::GPSLatitudeRef(clean_s(value)));
            }
            if let Some(value) = map
                .get("GPSLongitude")
                .and_then(|value| parse_sidecar_gps_coordinate(value))
            {
                metadata.set_tag(ExifTag::GPSLongitude(value));
            }
            if let Some(value) = map.get("GPSLongitudeRef") {
                metadata.set_tag(ExifTag::GPSLongitudeRef(clean_s(value)));
            }
            if let Some(value) = map.get("GPSAltitude")
                && let Some(value) = value
                    .split(|character: char| !(character.is_ascii_digit() || character == '.'))
                    .find(|part| !part.is_empty())
                    .and_then(|part| part.parse::<f64>().ok())
                    .and_then(decimal_to_ur64)
            {
                metadata.set_tag(ExifTag::GPSAltitude(vec![value]));
            }
            if let Some(value) = map.get("GPSAltitudeRef")
                && let Ok(value) = clean_s(value).parse::<u8>()
            {
                metadata.set_tag(ExifTag::GPSAltitudeRef(vec![value]));
            }
        }
    }

    if keep_metadata
        && !source_read_success
        && let Ok(file) = std::fs::File::open(original_path)
    {
        let mut bufreader = std::io::BufReader::new(&file);
        let exifreader = exif::Reader::new();

        if let Ok(exif_obj) = exifreader.read_from_container(&mut bufreader) {
            source_read_success = true;

            let get_string_val = |field: &exif::Field| -> String {
                match &field.value {
                    exif::Value::Ascii(vec) => vec
                        .iter()
                        .map(|v| {
                            String::from_utf8_lossy(v)
                                .trim_matches(char::from(0))
                                .to_string()
                        })
                        .collect::<Vec<String>>()
                        .join(" "),
                    _ => field
                        .display_value()
                        .to_string()
                        .replace("\"", "")
                        .trim()
                        .to_string(),
                }
            };

            if let Some(f) = exif_obj.get_field(exif::Tag::Make, exif::In::PRIMARY) {
                metadata.set_tag(ExifTag::Make(get_string_val(f)));
            }
            if let Some(f) = exif_obj.get_field(exif::Tag::Model, exif::In::PRIMARY) {
                metadata.set_tag(ExifTag::Model(get_string_val(f)));
            }
            if let Some(f) = exif_obj.get_field(exif::Tag::LensMake, exif::In::PRIMARY) {
                metadata.set_tag(ExifTag::LensMake(get_string_val(f)));
            }
            if let Some(f) = exif_obj.get_field(exif::Tag::LensModel, exif::In::PRIMARY) {
                metadata.set_tag(ExifTag::LensModel(get_string_val(f)));
            }
            if let Some(f) = exif_obj.get_field(exif::Tag::Artist, exif::In::PRIMARY) {
                metadata.set_tag(ExifTag::Artist(get_string_val(f)));
            }
            if let Some(f) = exif_obj.get_field(exif::Tag::Copyright, exif::In::PRIMARY) {
                metadata.set_tag(ExifTag::Copyright(get_string_val(f)));
            }
            if let Some(f) = exif_obj.get_field(exif::Tag::ImageDescription, exif::In::PRIMARY) {
                metadata.set_tag(ExifTag::ImageDescription(get_string_val(f)));
            }
            if let Some(f) = exif_obj.get_field(exif::Tag::UserComment, exif::In::PRIMARY) {
                let value = match &f.value {
                    exif::Value::Undefined(bytes, _) => {
                        let has_character_code = bytes.starts_with(b"ASCII\0\0\0")
                            || bytes.starts_with(b"UNICODE\0")
                            || bytes.starts_with(b"JIS\0\0\0\0\0");
                        let payload = if has_character_code {
                            &bytes[8..]
                        } else {
                            bytes
                        };
                        String::from_utf8_lossy(payload)
                            .trim_matches(char::from(0))
                            .trim()
                            .to_string()
                    }
                    _ => get_string_val(f),
                };
                if !value.is_empty() {
                    metadata.set_tag(ExifTag::UserComment(value.into_bytes()));
                }
            }
            if let Some(f) = exif_obj.get_field(exif::Tag::DateTimeOriginal, exif::In::PRIMARY) {
                metadata.set_tag(ExifTag::DateTimeOriginal(get_string_val(f)));
            }
            if let Some(f) = exif_obj.get_field(exif::Tag::DateTime, exif::In::PRIMARY) {
                metadata.set_tag(ExifTag::CreateDate(get_string_val(f)));
            }
            if let Some(f) = exif_obj.get_field(exif::Tag::FNumber, exif::In::PRIMARY)
                && let exif::Value::Rational(v) = &f.value
                && !v.is_empty()
            {
                metadata.set_tag(ExifTag::FNumber(vec![to_ur64(&v[0])]));
            }
            if let Some(f) = exif_obj.get_field(exif::Tag::ExposureTime, exif::In::PRIMARY)
                && let exif::Value::Rational(v) = &f.value
                && !v.is_empty()
            {
                metadata.set_tag(ExifTag::ExposureTime(vec![to_ur64(&v[0])]));
            }
            if let Some(f) = exif_obj.get_field(exif::Tag::FocalLength, exif::In::PRIMARY)
                && let exif::Value::Rational(v) = &f.value
                && !v.is_empty()
            {
                metadata.set_tag(ExifTag::FocalLength(vec![to_ur64(&v[0])]));
            }
            if let Some(f) = exif_obj.get_field(exif::Tag::ExposureBiasValue, exif::In::PRIMARY) {
                match &f.value {
                    exif::Value::SRational(v) if !v.is_empty() => {
                        metadata.set_tag(ExifTag::ExposureCompensation(vec![to_ir64(&v[0])]));
                    }
                    exif::Value::Rational(v) if !v.is_empty() => {
                        metadata.set_tag(ExifTag::ExposureCompensation(vec![iR64 {
                            nominator: v[0].num as i32,
                            denominator: v[0].denom as i32,
                        }]));
                    }
                    _ => {}
                }
            }
            if let Some(f) =
                exif_obj.get_field(exif::Tag::PhotographicSensitivity, exif::In::PRIMARY)
            {
                if let Some(val) = f.value.get_uint(0) {
                    metadata.set_tag(ExifTag::ISO(vec![val as u16]));
                }
            } else if let Some(f) = exif_obj.get_field(exif::Tag::ISOSpeed, exif::In::PRIMARY)
                && let Some(val) = f.value.get_uint(0)
            {
                metadata.set_tag(ExifTag::ISO(vec![val as u16]));
            }
            if let Some(f) = exif_obj.get_field(exif::Tag::FocalLengthIn35mmFilm, exif::In::PRIMARY)
                && let Some(val) = f.value.get_uint(0)
            {
                metadata.set_tag(ExifTag::FocalLengthIn35mmFormat(vec![val as u16]));
            }
            if !strip_gps {
                if let Some(f) = exif_obj.get_field(exif::Tag::GPSLatitude, exif::In::PRIMARY)
                    && let exif::Value::Rational(v) = &f.value
                    && v.len() >= 3
                {
                    metadata.set_tag(ExifTag::GPSLatitude(vec![
                        to_ur64(&v[0]),
                        to_ur64(&v[1]),
                        to_ur64(&v[2]),
                    ]));
                }
                if let Some(f) = exif_obj.get_field(exif::Tag::GPSLatitudeRef, exif::In::PRIMARY) {
                    metadata.set_tag(ExifTag::GPSLatitudeRef(get_string_val(f)));
                }
                if let Some(f) = exif_obj.get_field(exif::Tag::GPSLongitude, exif::In::PRIMARY)
                    && let exif::Value::Rational(v) = &f.value
                    && v.len() >= 3
                {
                    metadata.set_tag(ExifTag::GPSLongitude(vec![
                        to_ur64(&v[0]),
                        to_ur64(&v[1]),
                        to_ur64(&v[2]),
                    ]));
                }
                if let Some(f) = exif_obj.get_field(exif::Tag::GPSLongitudeRef, exif::In::PRIMARY) {
                    metadata.set_tag(ExifTag::GPSLongitudeRef(get_string_val(f)));
                }
                if let Some(f) = exif_obj.get_field(exif::Tag::GPSAltitude, exif::In::PRIMARY)
                    && let exif::Value::Rational(v) = &f.value
                    && !v.is_empty()
                {
                    metadata.set_tag(ExifTag::GPSAltitude(vec![to_ur64(&v[0])]));
                }
                if let Some(f) = exif_obj.get_field(exif::Tag::GPSAltitudeRef, exif::In::PRIMARY) {
                    let alt_ref = f.value.get_uint(0).unwrap_or(0) as u8;
                    metadata.set_tag(ExifTag::GPSAltitudeRef(vec![alt_ref]));
                }
            }
        }
    }

    if keep_metadata && !source_read_success && is_raw_file(original_path_str) {
        let loader = rawler::RawLoader::new();
        if let Ok(raw_source) = rawler::rawsource::RawSource::new(Path::new(original_path_str))
            && let Ok(decoder) = loader.get_decoder(&raw_source)
            && let Ok(meta) = decoder.raw_metadata(&raw_source, &Default::default())
        {
            if !meta.make.is_empty() {
                metadata.set_tag(ExifTag::Make(meta.make.clone()));
            }
            if !meta.model.is_empty() {
                metadata.set_tag(ExifTag::Model(meta.model.clone()));
            }
            let exif = meta.exif;
            if let Some(artist) = exif.artist {
                metadata.set_tag(ExifTag::Artist(artist));
            }
            if let Some(copyright) = exif.copyright {
                metadata.set_tag(ExifTag::Copyright(copyright));
            }
            if let Some(dt) = exif.date_time_original {
                metadata.set_tag(ExifTag::DateTimeOriginal(dt));
            }
            if let Some(dt) = exif.create_date {
                metadata.set_tag(ExifTag::CreateDate(dt));
            }
            if let Some(lens_make) = exif.lens_make {
                metadata.set_tag(ExifTag::LensMake(lens_make));
            }
            if let Some(lens_model) = exif.lens_model {
                metadata.set_tag(ExifTag::LensModel(lens_model));
            }
            if let Some(f) = exif.fnumber {
                metadata.set_tag(ExifTag::FNumber(vec![uR64 {
                    nominator: f.n,
                    denominator: f.d,
                }]));
            }
            if let Some(t) = exif.exposure_time {
                metadata.set_tag(ExifTag::ExposureTime(vec![uR64 {
                    nominator: t.n,
                    denominator: t.d,
                }]));
            }
            if let Some(fl) = exif.focal_length {
                metadata.set_tag(ExifTag::FocalLength(vec![uR64 {
                    nominator: fl.n,
                    denominator: fl.d,
                }]));
            }
            if let Some(iso) = exif.iso_speed {
                metadata.set_tag(ExifTag::ISO(vec![iso as u16]));
            } else if let Some(iso) = exif.iso_speed_ratings {
                metadata.set_tag(ExifTag::ISO(vec![iso]));
            }
            if let Some(ev) = exif.exposure_bias {
                metadata.set_tag(ExifTag::ExposureCompensation(vec![iR64 {
                    nominator: ev.n,
                    denominator: ev.d,
                }]));
            }
            if let Some(flash) = exif.flash {
                metadata.set_tag(ExifTag::Flash(vec![flash]));
            }
            if let Some(metering) = exif.metering_mode {
                metadata.set_tag(ExifTag::MeteringMode(vec![metering]));
            }
            if let Some(wb) = exif.white_balance {
                metadata.set_tag(ExifTag::WhiteBalance(vec![wb]));
            }
            if let Some(prog) = exif.exposure_program {
                metadata.set_tag(ExifTag::ExposureProgram(vec![prog]));
            }
            if !strip_gps && let Some(gps) = exif.gps {
                if let Some(lat) = gps.gps_latitude {
                    metadata.set_tag(ExifTag::GPSLatitude(vec![
                        uR64 {
                            nominator: lat[0].n,
                            denominator: lat[0].d,
                        },
                        uR64 {
                            nominator: lat[1].n,
                            denominator: lat[1].d,
                        },
                        uR64 {
                            nominator: lat[2].n,
                            denominator: lat[2].d,
                        },
                    ]));
                }
                if let Some(lat_ref) = gps.gps_latitude_ref {
                    metadata.set_tag(ExifTag::GPSLatitudeRef(lat_ref));
                }
                if let Some(lon) = gps.gps_longitude {
                    metadata.set_tag(ExifTag::GPSLongitude(vec![
                        uR64 {
                            nominator: lon[0].n,
                            denominator: lon[0].d,
                        },
                        uR64 {
                            nominator: lon[1].n,
                            denominator: lon[1].d,
                        },
                        uR64 {
                            nominator: lon[2].n,
                            denominator: lon[2].d,
                        },
                    ]));
                }
                if let Some(lon_ref) = gps.gps_longitude_ref {
                    metadata.set_tag(ExifTag::GPSLongitudeRef(lon_ref));
                }
                if let Some(alt) = gps.gps_altitude {
                    metadata.set_tag(ExifTag::GPSAltitude(vec![uR64 {
                        nominator: alt.n,
                        denominator: alt.d,
                    }]));
                }
                if let Some(alt_ref) = gps.gps_altitude_ref {
                    metadata.set_tag(ExifTag::GPSAltitudeRef(vec![alt_ref]));
                }
            }
        }
    }

    if let Some(overrides) = overrides {
        if overrides.artist.is_some() {
            if let Some(value) = ExportMetadataOverrides::normalized(&overrides.artist, 512) {
                metadata.set_tag(ExifTag::Artist(value));
            } else {
                metadata.remove_tag(ExifTag::Artist(String::new()));
            }
        }
        if overrides.copyright.is_some() {
            if let Some(value) = ExportMetadataOverrides::normalized(&overrides.copyright, 512) {
                metadata.set_tag(ExifTag::Copyright(value));
            } else {
                metadata.remove_tag(ExifTag::Copyright(String::new()));
            }
        }
        if overrides.description.is_some() {
            if let Some(value) = ExportMetadataOverrides::normalized(&overrides.description, 2_048)
            {
                metadata.set_tag(ExifTag::ImageDescription(value));
            } else {
                metadata.remove_tag(ExifTag::ImageDescription(String::new()));
            }
        }
        if overrides.contact.is_some() {
            if let Some(value) = ExportMetadataOverrides::normalized(&overrides.contact, 512) {
                metadata.set_tag(ExifTag::UserComment(
                    format!("Contact: {value}").into_bytes(),
                ));
            } else {
                metadata.remove_tag(ExifTag::UserComment(Vec::new()));
            }
        }
    }

    metadata.set_tag(ExifTag::Software("RAW Editor".to_string()));
    metadata.set_tag(ExifTag::Orientation(vec![1u16]));
    metadata.set_tag(ExifTag::ColorSpace(vec![1u16]));

    Ok(Some((metadata, file_type)))
}

pub(crate) fn export_metadata_tiff_payload(
    original_path_str: &str,
    output_format: &str,
    keep_metadata: bool,
    strip_gps: bool,
    overrides: Option<&ExportMetadataOverrides>,
) -> Result<Option<Vec<u8>>, String> {
    let Some((metadata, _)) = build_export_metadata(
        original_path_str,
        output_format,
        keep_metadata,
        strip_gps,
        overrides,
    )?
    else {
        return Ok(None);
    };

    let encoded = metadata
        .as_u8_vec(FileExtension::JPEG)
        .map_err(|error| format!("Failed to encode export metadata: {error}"))?;
    const JPEG_EXIF_PREFIX: &[u8] = b"\xff\xe1";
    const EXIF_HEADER: &[u8] = b"Exif\0\0";
    const TIFF_PAYLOAD_OFFSET: usize = 10;
    if encoded.len() < TIFF_PAYLOAD_OFFSET
        || &encoded[..2] != JPEG_EXIF_PREFIX
        || &encoded[4..TIFF_PAYLOAD_OFFSET] != EXIF_HEADER
    {
        return Err("Metadata encoder returned an invalid JPEG EXIF segment".to_string());
    }
    let declared_segment_len = u16::from_be_bytes([encoded[2], encoded[3]]) as usize;
    if declared_segment_len + 2 != encoded.len() {
        return Err(format!(
            "Encoded EXIF segment is too large or malformed ({} bytes)",
            encoded.len()
        ));
    }

    Ok(Some(encoded[TIFF_PAYLOAD_OFFSET..].to_vec()))
}

#[cfg(not(target_os = "android"))]
pub(crate) fn export_metadata_for_streaming_tiff(
    original_path_str: &str,
    keep_metadata: bool,
    strip_gps: bool,
    overrides: Option<&ExportMetadataOverrides>,
) -> Result<Option<Metadata>, String> {
    Ok(build_export_metadata(
        original_path_str,
        "tiff",
        keep_metadata,
        strip_gps,
        overrides,
    )?
    .map(|(metadata, _)| metadata))
}

pub fn write_image_with_metadata(
    image_bytes: &mut Vec<u8>,
    original_path_str: &str,
    output_format: &str,
    keep_metadata: bool,
    strip_gps: bool,
    overrides: Option<&ExportMetadataOverrides>,
) -> Result<(), String> {
    // Desktop TIFF exports attach selected metadata while the streaming encoder is still writing
    // the file. Avoid little_exif's whole-file TIFF rewrite on legacy/full-frame paths.
    if matches!(output_format.to_ascii_lowercase().as_str(), "tif" | "tiff") {
        return Ok(());
    }

    let Some((metadata, file_type)) = build_export_metadata(
        original_path_str,
        output_format,
        keep_metadata,
        strip_gps,
        overrides,
    )?
    else {
        return Ok(());
    };

    if let Err(e) = metadata.write_to_vec(image_bytes, file_type) {
        log::warn!("Failed to write metadata: {}", e);
    }

    Ok(())
}
