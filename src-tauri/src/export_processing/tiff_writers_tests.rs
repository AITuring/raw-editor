//! Inspect the real IFD and pixels for both production TIFF container kinds.

use super::*;
use std::io::{Cursor, Read, Seek, SeekFrom};
use tiff::decoder::{Decoder, DecodingResult};

fn encoded_bytes(mut file: fs::File) -> Vec<u8> {
    file.seek(SeekFrom::Start(0)).unwrap();
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).unwrap();
    bytes
}

fn checked_decoder(
    bytes: Vec<u8>,
    bit_depth: u16,
    has_alpha: bool,
    has_icc: bool,
    bigtiff: bool,
) -> Decoder<Cursor<Vec<u8>>> {
    let version = if &bytes[..2] == b"II" {
        u16::from_le_bytes([bytes[2], bytes[3]])
    } else {
        assert_eq!(&bytes[..2], b"MM");
        u16::from_be_bytes([bytes[2], bytes[3]])
    };
    assert_eq!(version, if bigtiff { 43 } else { 42 });
    let mut decoder = Decoder::new(Cursor::new(bytes)).unwrap();
    assert_eq!(decoder.dimensions().unwrap(), (3, 2));
    let samples = if has_alpha { 4 } else { 3 };
    assert_eq!(
        decoder.get_tag_u32(TiffTag::SamplesPerPixel).unwrap(),
        samples
    );
    assert_eq!(
        decoder.get_tag_u16_vec(TiffTag::BitsPerSample).unwrap(),
        vec![bit_depth; samples as usize]
    );
    assert_eq!(
        decoder
            .get_tag_u32(TiffTag::PhotometricInterpretation)
            .unwrap(),
        2
    );
    if has_alpha {
        assert_eq!(decoder.get_tag_u16_vec(TiffTag::ExtraSamples).unwrap(), [2]);
    } else {
        assert!(decoder.find_tag(TiffTag::ExtraSamples).unwrap().is_none());
    }
    if has_icc {
        assert_eq!(
            decoder.get_tag_u8_vec(TiffTag::IccProfile).unwrap(),
            srgb_v4_profile()
        );
    } else {
        assert!(decoder.find_tag(TiffTag::IccProfile).unwrap().is_none());
    }
    decoder
}

#[test]
fn tiff_writers_rgba16_declares_alpha_and_preserves_samples_in_both_containers() {
    let rgba = [
        64001, 12345, 37, 0, 9, 58017, 4123, 32768, 33001, 70, 65211, 65535, 71, 25000, 199, 0,
        1081, 17891, 64311, 1, 65535, 0, 32767, 65534,
    ];
    for bigtiff in [false, true] {
        for has_alpha in [false, true] {
            for has_icc in [false, true] {
                let data = rgba
                    .chunks_exact(4)
                    .flat_map(|p| p[..if has_alpha { 4 } else { 3 }].iter().copied())
                    .collect::<Vec<u16>>();
                let mut output = tempfile::tempfile().unwrap();
                match (bigtiff, has_alpha) {
                    (false, false) => {
                        encode_rgb16_tiff_with_metadata(&mut output, 3, 2, &data, has_icc, None)
                    }
                    (false, true) => {
                        encode_rgba16_tiff_with_metadata(&mut output, 3, 2, &data, has_icc, None)
                    }
                    (true, false) => {
                        encode_rgb16_bigtiff_with_metadata(&mut output, 3, 2, &data, has_icc, None)
                    }
                    (true, true) => {
                        encode_rgba16_bigtiff_with_metadata(&mut output, 3, 2, &data, has_icc, None)
                    }
                }
                .unwrap();
                let mut decoder =
                    checked_decoder(encoded_bytes(output), 16, has_alpha, has_icc, bigtiff);
                let DecodingResult::U16(decoded) = decoder.read_image().unwrap() else {
                    panic!("16-bit writer changed sample representation");
                };
                // Includes nonzero RGB beneath alpha=0 and partial-alpha
                // samples: passing requires no premultiplication or clipping.
                assert_eq!(decoded, data);
            }
        }
    }
}

#[test]
fn tiff_writers_rgba8_declares_alpha_and_preserves_samples_in_both_containers() {
    let rgba = [
        201, 17, 35, 0, 9, 203, 41, 128, 33, 7, 251, 255, 71, 25, 199, 0, 10, 178, 231, 1, 255, 0,
        127, 254,
    ];
    for bigtiff in [false, true] {
        for has_alpha in [false, true] {
            for has_icc in [false, true] {
                let data = rgba
                    .chunks_exact(4)
                    .flat_map(|p| p[..if has_alpha { 4 } else { 3 }].iter().copied())
                    .collect::<Vec<u8>>();
                let mut output = tempfile::tempfile().unwrap();
                match (bigtiff, has_alpha) {
                    (false, false) => {
                        encode_rgb8_tiff_with_metadata(&mut output, 3, 2, &data, has_icc, None)
                    }
                    (false, true) => {
                        encode_rgba8_tiff_with_metadata(&mut output, 3, 2, &data, has_icc, None)
                    }
                    (true, false) => {
                        encode_rgb8_bigtiff_with_metadata(&mut output, 3, 2, &data, has_icc, None)
                    }
                    (true, true) => {
                        encode_rgba8_bigtiff_with_metadata(&mut output, 3, 2, &data, has_icc, None)
                    }
                }
                .unwrap();
                let mut decoder =
                    checked_decoder(encoded_bytes(output), 8, has_alpha, has_icc, bigtiff);
                let DecodingResult::U8(decoded) = decoder.read_image().unwrap() else {
                    panic!("8-bit writer changed sample representation");
                };
                assert_eq!(decoded, data);
            }
        }
    }
}

#[test]
fn tiff_writers_alpha_declaration_preserves_standard_metadata() {
    use little_exif::exif_tag::ExifTag;
    let mut metadata = ExifMetadata::new();
    metadata.set_tag(ExifTag::Artist("Coverage fixture".into()));
    metadata.set_tag(ExifTag::DateTimeOriginal("2026:11:02 12:00:00".into()));
    let mut output = tempfile::tempfile().unwrap();
    encode_rgba16_tiff_with_metadata(&mut output, 3, 2, &[127; 24], true, Some(&metadata)).unwrap();
    let mut decoder = checked_decoder(encoded_bytes(output), 16, true, true, false);
    assert_eq!(
        decoder
            .get_tag_ascii_string(TiffTag::Artist)
            .unwrap()
            .trim_end_matches('\0'),
        "Coverage fixture"
    );
    assert!(decoder.get_tag_u64(TiffTag::ExifDirectory).unwrap() > 0);
}

#[test]
fn tiff_writers_rgba_rejects_incorrect_sample_count() {
    for bigtiff in [false, true] {
        let mut output = tempfile::tempfile().unwrap();
        let result = if bigtiff {
            encode_rgba16_bigtiff_with_metadata(&mut output, 3, 2, &[0; 30], true, None)
        } else {
            encode_rgba16_tiff_with_metadata(&mut output, 3, 2, &[0; 18], true, None)
        };
        assert!(result.unwrap_err().contains("expected 24"));
    }
}
