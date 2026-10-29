//! EXIF and TIFF metadata in streamed exports.
//!
//! Streamed JPEG, PNG and TIFF keep bounded source metadata, honour cleared fields
//! and strip GPS on request.

use super::*;

fn encode_streaming_fixture_with_exif(
    image: &DynamicImage,
    output_format: &str,
    jpeg_quality: u8,
    export_exif: Option<&[u8]>,
) -> Result<Vec<u8>, String> {
    encode_streaming_fixture_with_metadata(
        image,
        output_format,
        jpeg_quality,
        export_exif.map_or(StreamingExportMetadata::None, |payload| {
            StreamingExportMetadata::ExifTiffPayload(payload)
        }),
    )
}

fn encode_streaming_fixture_with_tiff_metadata(
    image: &DynamicImage,
    export_metadata: Option<&Metadata>,
) -> Result<Vec<u8>, String> {
    encode_streaming_fixture_with_metadata(
        image,
        "tiff",
        100,
        export_metadata.map_or(StreamingExportMetadata::None, |metadata| {
            StreamingExportMetadata::TiffDirectories(metadata)
        }),
    )
}

#[test]
fn streaming_jpeg_png_and_tiff_embed_bounded_exif_and_strip_gps() {
    fn read_export_exif(encoded: &[u8]) -> exif::Exif {
        exif::Reader::new()
            .read_from_container(&mut Cursor::new(encoded))
            .expect("read streamed export EXIF")
    }

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

    let image = fixture_image(37, 23);
    let mut source = encode_image_to_bytes(&image, "jpeg", 92).expect("encode EXIF source");
    let mut source_metadata = Metadata::new();
    source_metadata.set_tag(ExifTag::Make("Bounded Camera".to_string()));
    source_metadata.set_tag(ExifTag::Model("Synthetic 60MP".to_string()));
    source_metadata.set_tag(ExifTag::Artist("Original Artist".to_string()));
    source_metadata.set_tag(ExifTag::ImageDescription(
        "Museum export fixture".to_string(),
    ));
    source_metadata.set_tag(ExifTag::DateTimeOriginal("2026:08:10 12:34:56".to_string()));
    source_metadata.set_tag(ExifTag::GPSLatitudeRef("N".to_string()));
    source_metadata.set_tag(ExifTag::GPSLatitude(vec![
        uR64 {
            nominator: 31,
            denominator: 1,
        },
        uR64 {
            nominator: 14,
            denominator: 1,
        },
        uR64 {
            nominator: 15,
            denominator: 1,
        },
    ]));
    source_metadata
        .write_to_vec(&mut source, FileExtension::JPEG)
        .expect("attach source EXIF");

    let directory = tempfile::tempdir().expect("create metadata test directory");
    let source_path = directory.path().join("source.jpg");
    std::fs::write(&source_path, source).expect("write EXIF source");
    let source_path = source_path.to_string_lossy();

    let retained =
        exif_processing::export_metadata_tiff_payload(&source_path, "jpeg", true, false, None)
            .expect("build retained EXIF")
            .expect("retained EXIF must exist");
    let jpeg = encode_streaming_fixture_with_exif(&image, "jpeg", 92, Some(&retained))
        .expect("stream JPEG with EXIF");
    assert_decoder_contract(
        JpegDecoder::new(Cursor::new(&jpeg)).expect("decode metadata JPEG"),
        (37, 23),
        true,
    );
    let jpeg_exif = read_export_exif(&jpeg);
    assert_eq!(
        ascii_tag(&jpeg_exif, exif::Tag::Make).as_deref(),
        Some("Bounded Camera")
    );
    assert_eq!(
        ascii_tag(&jpeg_exif, exif::Tag::ImageDescription).as_deref(),
        Some("Museum export fixture")
    );
    assert!(
        jpeg_exif
            .get_field(exif::Tag::GPSLatitude, exif::In::PRIMARY)
            .is_some(),
        "JPEG must retain GPS when stripGps is false"
    );

    let cleared_metadata = exif_processing::export_metadata_tiff_payload(
        &source_path,
        "jpeg",
        true,
        false,
        Some(&exif_processing::ExportMetadataOverrides {
            artist: Some(String::new()),
            contact: None,
            copyright: None,
            description: None,
        }),
    )
    .expect("build EXIF with an explicit cleared artist")
    .expect("cleared EXIF must still contain retained metadata");
    let cleared_jpeg =
        encode_streaming_fixture_with_exif(&image, "jpeg", 92, Some(&cleared_metadata))
            .expect("stream JPEG with cleared artist");
    assert!(
        read_export_exif(&cleared_jpeg)
            .get_field(exif::Tag::Artist, exif::In::PRIMARY)
            .is_none(),
        "an empty editable artist field must remove the retained source artist"
    );

    let stripped =
        exif_processing::export_metadata_tiff_payload(&source_path, "png", true, true, None)
            .expect("build stripped EXIF")
            .expect("stripped EXIF must exist");
    let png = encode_streaming_fixture_with_exif(&image, "png", 92, Some(&stripped))
        .expect("stream PNG with EXIF");
    assert_decoder_contract(
        PngDecoder::new(Cursor::new(&png)).expect("decode metadata PNG"),
        (37, 23),
        true,
    );
    let png_exif = read_export_exif(&png);
    assert_eq!(
        ascii_tag(&png_exif, exif::Tag::Make).as_deref(),
        Some("Bounded Camera")
    );
    assert_eq!(
        ascii_tag(&png_exif, exif::Tag::Software).as_deref(),
        Some("RAW Editor")
    );
    assert!(
        png_exif
            .get_field(exif::Tag::GPSLatitude, exif::In::PRIMARY)
            .is_none(),
        "PNG must remove GPS when stripGps is true"
    );

    let retained_tiff_metadata =
        exif_processing::export_metadata_for_streaming_tiff(&source_path, true, false, None)
            .expect("build retained TIFF metadata")
            .expect("retained TIFF metadata must exist");
    let tiff = encode_streaming_fixture_with_tiff_metadata(&image, Some(&retained_tiff_metadata))
        .expect("stream TIFF with metadata");
    assert_decoder_contract(
        TiffDecoder::new(Cursor::new(&tiff)).expect("decode metadata TIFF"),
        (37, 23),
        true,
    );
    assert_eq!(
        image::load_from_memory_with_format(&tiff, ImageFormat::Tiff)
            .expect("load metadata TIFF pixels")
            .to_rgba8(),
        image.to_rgba8()
    );
    let tiff_exif = read_export_exif(&tiff);
    assert_eq!(
        ascii_tag(&tiff_exif, exif::Tag::Make).as_deref(),
        Some("Bounded Camera")
    );
    assert_eq!(
        ascii_tag(&tiff_exif, exif::Tag::Software).as_deref(),
        Some("RAW Editor")
    );
    assert_eq!(
        ascii_tag(&tiff_exif, exif::Tag::DateTimeOriginal).as_deref(),
        Some("2026:08:10 12:34:56")
    );
    assert!(
        tiff_exif
            .get_field(exif::Tag::GPSLatitude, exif::In::PRIMARY)
            .is_some(),
        "TIFF must retain GPS when stripGps is false"
    );

    let tiff_source_path = directory.path().join("source.tiff");
    std::fs::write(&tiff_source_path, &tiff).expect("write metadata TIFF source");
    let copied_tiff_metadata = exif_processing::export_metadata_for_streaming_tiff(
        &tiff_source_path.to_string_lossy(),
        true,
        false,
        None,
    )
    .expect("copy TIFF source metadata")
    .expect("copied TIFF metadata must exist");
    let copied_tiff =
        encode_streaming_fixture_with_tiff_metadata(&image, Some(&copied_tiff_metadata))
            .expect("stream TIFF copied from TIFF metadata");
    let copied_tiff_exif = read_export_exif(&copied_tiff);
    assert_eq!(
        ascii_tag(&copied_tiff_exif, exif::Tag::Make).as_deref(),
        Some("Bounded Camera")
    );
    assert!(
        copied_tiff_exif
            .get_field(exif::Tag::GPSLatitude, exif::In::PRIMARY)
            .is_some(),
        "TIFF source metadata must remain readable and copyable"
    );

    let stripped_tiff_metadata =
        exif_processing::export_metadata_for_streaming_tiff(&source_path, true, true, None)
            .expect("build stripped TIFF metadata")
            .expect("stripped TIFF metadata must exist");
    let stripped_tiff =
        encode_streaming_fixture_with_tiff_metadata(&image, Some(&stripped_tiff_metadata))
            .expect("stream TIFF without GPS");
    let stripped_tiff_exif = read_export_exif(&stripped_tiff);
    assert_eq!(
        ascii_tag(&stripped_tiff_exif, exif::Tag::Make).as_deref(),
        Some("Bounded Camera")
    );
    assert!(
        stripped_tiff_exif
            .get_field(exif::Tag::GPSLatitude, exif::In::PRIMARY)
            .is_none(),
        "TIFF must remove GPS when stripGps is true"
    );
}
