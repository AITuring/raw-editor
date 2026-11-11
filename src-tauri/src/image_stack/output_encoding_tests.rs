use super::*;
use image::{ImageBuffer, Rgba};
use std::io::Cursor;
use tiff::decoder::{Decoder, DecodingResult};

#[test]
fn generic_tiff_writer_declares_alpha_and_preserves_pixels_profile_and_rgb_only() {
    let source = DynamicImage::ImageRgba16(ImageBuffer::from_fn(3, 2, |x, y| {
        Rgba([
            64001 - x as u16 * 1000,
            12345 + y as u16 * 700,
            37 + x as u16,
            [0, 32768, 65535][x as usize],
        ])
    }));
    for bits in [8, 16] {
        for has_alpha in [false, true] {
            for has_icc in [false, true] {
                let image = if has_alpha {
                    source.clone()
                } else {
                    DynamicImage::ImageRgb16(source.to_rgb16())
                };
                let mut bytes = Cursor::new(Vec::new());
                encode_srgb_image_stack(
                    &image,
                    &mut bytes,
                    ImageStackOutputFormat::Tiff,
                    95,
                    bits,
                    has_icc,
                    None,
                )
                .unwrap();
                bytes.set_position(0);
                let mut decoder = Decoder::new(bytes).unwrap();
                let sample_count = if has_alpha { 4 } else { 3 };
                assert_eq!(decoder.dimensions().unwrap(), (3, 2));
                assert_eq!(
                    decoder.get_tag_u32(Tag::SamplesPerPixel).unwrap(),
                    sample_count
                );
                assert_eq!(
                    decoder.get_tag_u16_vec(Tag::BitsPerSample).unwrap(),
                    vec![u16::from(bits); sample_count as usize]
                );
                if has_alpha {
                    assert_eq!(decoder.get_tag_u16_vec(Tag::ExtraSamples).unwrap(), [2]);
                } else {
                    assert!(decoder.find_tag(Tag::ExtraSamples).unwrap().is_none());
                }
                if has_icc {
                    assert_eq!(
                        decoder.get_tag_u8_vec(Tag::IccProfile).unwrap(),
                        crate::color_management::srgb_v4_profile()
                    );
                } else {
                    assert!(decoder.find_tag(Tag::IccProfile).unwrap().is_none());
                }
                match decoder.read_image().unwrap() {
                    DecodingResult::U16(decoded) if bits == 16 => {
                        let expected = if has_alpha {
                            image.to_rgba16().into_raw()
                        } else {
                            image.to_rgb16().into_raw()
                        };
                        assert_eq!(decoded, expected);
                    }
                    DecodingResult::U8(decoded) if bits == 8 => {
                        let expected = if has_alpha {
                            image.to_rgba8().into_raw()
                        } else {
                            image.to_rgb8().into_raw()
                        };
                        assert_eq!(decoded, expected);
                    }
                    _ => panic!("TIFF sample type disagrees with requested bit depth"),
                }
            }
        }
    }
}
