#[cfg(test)]
use std::fs;

#[cfg(test)]
use crate::image_processing::ImageMetadata;
#[cfg(test)]
use crate::image_processing::SIDECAR_SCHEMA_VERSION;

mod sidecar;

pub use sidecar::{load_sidecar, load_sidecar_checked, save_sidecar, truncate_large_exif};
#[cfg(test)]
use sidecar::{read_sidecar_document, sidecar_backup_path};

mod exif_values;

#[cfg(test)]
use exif_values::format_exposure_time_seconds;
use exif_values::{
    clean_ascii_value, decimal_to_ur64, fmt_date_str, format_exif_exposure_time,
    format_lens_specification, format_min_max, parse_creation_datetime, parse_creation_field,
    parse_raw_creation_date, parse_sidecar_gps_coordinate, rational_to_f32_checked,
    rawler_rational_to_f32_checked, to_ir64, to_ur64,
};

mod metadata_read;

#[cfg(target_os = "android")]
pub use metadata_read::get_creation_date_from_bytes;
pub use metadata_read::{
    extract_metadata, get_creation_date_from_path, read_exif, read_exposure_time_secs, read_iso,
    read_raw_metadata, try_get_exif_creation_date,
};

mod export_metadata;

#[cfg(not(target_os = "android"))]
pub(crate) use export_metadata::export_metadata_for_streaming_tiff;
pub(crate) use export_metadata::export_metadata_tiff_payload;
pub use export_metadata::{ExportMetadataOverrides, write_image_with_metadata};

mod exif_sidecar;

pub use exif_sidecar::{
    focal_length_35mm_from_bytes, get_primary_sidecar_path, get_rrexif_path,
    persist_exif_if_missing, read_exif_data, read_exif_data_from_bytes, read_rrexif_sidecar,
    write_rrexif_sidecar,
};

#[cfg(test)]
mod tests;
