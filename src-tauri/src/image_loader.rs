use crate::app_settings::load_settings;
use crate::app_state::{AppState, LoadedImage};
use crate::exif_processing;
use crate::file_management::{parse_virtual_path, read_file_mapped};
use crate::formats::is_raw_file;
use crate::image_processing::ImageMetadata;
#[cfg(test)]
use crate::psd_loader::load_psd_image;
use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, atomic::Ordering};
#[derive(serde::Serialize)]
pub struct LoadImageResult {
    pub width: u32,
    pub height: u32,
    pub metadata: ImageMetadata,
    pub exif: HashMap<String, String>,
    pub is_raw: bool,
}

mod source_decode;

use source_decode::load_base_image_from_bytes_with_exif_persistence;
#[cfg(test)]
use source_decode::read_embedded_icc;
pub use source_decode::{
    load_base_image_from_bytes, load_base_image_from_bytes_without_exif_persistence,
    load_image_with_orientation,
};

mod patch_compositing;

pub use patch_compositing::{composite_patches_on_image, load_and_composite};

#[tauri::command]
pub fn is_image_cached(path: String, state: tauri::State<'_, AppState>) -> bool {
    let (source_path, _) = parse_virtual_path(&path);
    let source_path_str = source_path.to_string_lossy().to_string();
    state
        .decoded_image_cache
        .lock()
        .unwrap()
        .get(&source_path_str)
        .is_some()
}

#[tauri::command]
pub async fn load_image(
    path: String,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<LoadImageResult, String> {
    let my_generation = state.load_image_generation.fetch_add(1, Ordering::SeqCst) + 1;
    let generation_tracker = state.load_image_generation.clone();
    let cancel_token = Some((generation_tracker.clone(), my_generation));

    {
        *state.original_image.lock().unwrap() = None;
        *state.cached_preview.lock().unwrap() = None;
        *state.gpu_image_cache.lock().unwrap() = None;
        *state.full_warped_cache.lock().unwrap() = None;
        *state.range_mask_source_cache.lock().unwrap() = None;
        *state.full_transformed_cache.lock().unwrap() = None;

        state.mask_cache.lock().unwrap().clear();
        state.patch_cache.lock().unwrap().clear();
        state.geometry_cache.lock().unwrap().clear();

        *state.denoise_result.lock().unwrap() = None;
        *state.hdr_result.lock().unwrap() = None;
        *state.panorama_result.lock().unwrap() = None;
    }

    let (source_path, sidecar_path) = parse_virtual_path(&path);
    let source_path_str = source_path.to_string_lossy().to_string();

    let metadata: ImageMetadata = crate::exif_processing::load_sidecar(&sidecar_path);

    let settings = load_settings(app_handle.clone()).unwrap_or_default();

    let path_clone = source_path_str.clone();

    let cached_data = state
        .decoded_image_cache
        .lock()
        .unwrap()
        .get(&source_path_str);

    let (pristine_arc, exif_data) = if let Some((cached_img, cached_exif)) = cached_data {
        (cached_img, cached_exif)
    } else {
        if crate::file_management::is_cloud_placeholder(&source_path) {
            return Err(format!(
                "'{}' is stored in iCloud and hasn't been downloaded yet. Download it in Finder, then try again.",
                source_path_str
            ));
        }

        let (pristine_img, exif_data_loaded) = tokio::task::spawn_blocking(move || {
            if generation_tracker.load(Ordering::SeqCst) != my_generation {
                return Err("Load cancelled".to_string());
            }

            let result: Result<(DynamicImage, HashMap<String, String>), String> =
                (|| match read_file_mapped(Path::new(&path_clone)) {
                    Ok(mmap) => {
                        if generation_tracker.load(Ordering::SeqCst) != my_generation {
                            return Err("Load cancelled".to_string());
                        }

                        let img = load_base_image_from_bytes(
                            &mmap,
                            &path_clone,
                            false,
                            &settings,
                            cancel_token.clone(),
                        )
                        .map_err(|e| e.to_string())?;
                        let exif = exif_processing::read_exif_data(&path_clone, &mmap);
                        Ok((img, exif))
                    }
                    Err(e) => {
                        log::warn!(
                            "Failed to memory-map file '{}': {}. Falling back to standard read.",
                            path_clone,
                            e
                        );
                        let bytes = fs::read(&path_clone).map_err(|io_err| {
                            format!("Fallback read failed for {}: {}", path_clone, io_err)
                        })?;

                        if generation_tracker.load(Ordering::SeqCst) != my_generation {
                            return Err("Load cancelled".to_string());
                        }

                        let img = load_base_image_from_bytes(
                            &bytes,
                            &path_clone,
                            false,
                            &settings,
                            cancel_token.clone(),
                        )
                        .map_err(|e| e.to_string())?;
                        let exif = exif_processing::read_exif_data(&path_clone, &bytes);
                        Ok((img, exif))
                    }
                })();
            result
        })
        .await
        .map_err(|e| e.to_string())??;

        let arc_img = Arc::new(pristine_img);

        state.decoded_image_cache.lock().unwrap().insert(
            source_path_str.clone(),
            arc_img.clone(),
            exif_data_loaded.clone(),
        );

        (arc_img, exif_data_loaded)
    };

    if state.load_image_generation.load(Ordering::SeqCst) != my_generation {
        return Err("Load cancelled".to_string());
    }

    let is_raw = is_raw_file(&source_path_str);

    if state.load_image_generation.load(Ordering::SeqCst) != my_generation {
        return Err("Load cancelled".to_string());
    }

    let (orig_width, orig_height) = pristine_arc.dimensions();

    *state.original_image.lock().unwrap() = Some(LoadedImage {
        path,
        image: pristine_arc,
        is_raw,
    });

    Ok(LoadImageResult {
        width: orig_width,
        height: orig_height,
        metadata,
        exif: exif_data,
        is_raw,
    })
}

#[cfg(test)]
mod color_profile_tests {
    use std::io::Cursor;

    use image::codecs::{jpeg::JpegEncoder, png::PngEncoder, tiff::TiffEncoder};
    use image::{DynamicImage, ExtendedColorType, ImageBuffer, ImageEncoder, ImageReader, Rgb};
    use moxcms::ColorProfile;

    use super::{load_image_with_orientation, load_psd_image, read_embedded_icc};
    use crate::color_management::srgb_v4_profile;

    fn encode_rgb_png(pixel: [u8; 3], profile: Option<Vec<u8>>) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut encoder = PngEncoder::new(&mut bytes);
        if let Some(profile) = profile {
            encoder
                .set_icc_profile(profile)
                .expect("attach deterministic test profile");
        }
        encoder
            .write_image(&pixel, 1, 1, ExtendedColorType::Rgb8)
            .expect("encode deterministic PNG");
        bytes
    }

    fn encode_rgb_jpeg(pixel: [u8; 3], profile: Vec<u8>) -> Vec<u8> {
        let pixels = pixel.repeat(16);
        let mut bytes = Vec::new();
        let mut encoder = JpegEncoder::new_with_quality(&mut bytes, 100);
        encoder
            .set_icc_profile(profile)
            .expect("attach deterministic test profile");
        encoder
            .write_image(&pixels, 4, 4, ExtendedColorType::Rgb8)
            .expect("encode deterministic JPEG");
        bytes
    }

    fn encode_rgb_tiff(pixel: [u8; 3], profile: Vec<u8>) -> Vec<u8> {
        let image = ImageBuffer::<Rgb<u16>, Vec<u16>>::from_pixel(
            37,
            23,
            Rgb([
                u16::from(pixel[0]) * 257,
                u16::from(pixel[1]) * 257,
                u16::from(pixel[2]) * 257,
            ]),
        );
        let mut bytes = Vec::new();
        let mut cursor = Cursor::new(&mut bytes);
        let mut encoder = TiffEncoder::new(&mut cursor);
        encoder
            .set_icc_profile(profile)
            .expect("attach deterministic test profile");
        DynamicImage::ImageRgb16(image)
            .write_with_encoder(encoder)
            .expect("encode deterministic TIFF");
        bytes
    }

    fn decoded_pixel(bytes: &[u8]) -> [f32; 3] {
        let decoded = load_image_with_orientation(bytes, None)
            .expect("decode test image")
            .into_rgb32f();
        decoded.get_pixel(0, 0).0
    }

    fn embedded_icc(bytes: &[u8]) -> Option<Vec<u8>> {
        let reader = ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .expect("guess test image format");
        let format = reader.format();
        let mut decoder = reader
            .into_decoder()
            .expect("initialize test image decoder");
        read_embedded_icc(bytes, format, &mut decoder).expect("read test image ICC")
    }

    fn assert_pixel_close(label: &str, actual: [f32; 3], expected: [f32; 3], tolerance: f32) {
        for channel in 0..3 {
            assert!(
                (actual[channel] - expected[channel]).abs() <= tolerance,
                "{label} channel {channel}: expected {}, got {}",
                expected[channel],
                actual[channel]
            );
        }
    }

    #[test]
    fn jpeg_png_and_tiff_display_p3_profiles_are_applied_during_decode() {
        let mut display_p3 = ColorProfile::new_display_p3();
        display_p3.description = None;
        display_p3.copyright = None;
        display_p3.cicp = None;
        let profile = display_p3.encode().expect("encode Display P3 profile");
        assert_eq!(profile.len() % 4, 0, "test ICC must be 4-byte aligned");
        let png = encode_rgb_png([128, 64, 191], Some(profile.clone()));
        let jpeg = encode_rgb_jpeg([128, 64, 191], profile.clone());
        let tiff = encode_rgb_tiff([128, 64, 191], profile);
        let srgb_tiff = encode_rgb_tiff([128, 64, 191], srgb_v4_profile().to_vec());
        assert_eq!(embedded_icc(&srgb_tiff).as_deref(), Some(srgb_v4_profile()));

        for (label, encoded) in [("PNG", &png), ("JPEG", &jpeg), ("TIFF", &tiff)] {
            assert!(
                embedded_icc(encoded).is_some(),
                "{label} must expose its ICC profile"
            );
        }

        for (label, encoded) in [("PNG", &png), ("JPEG", &jpeg), ("TIFF", &tiff)] {
            assert_pixel_close(label, decoded_pixel(encoded), [0.539, 0.233, 0.776], 0.008);
        }
    }

    #[test]
    fn untagged_or_malformed_profile_falls_back_to_srgb_assumption() {
        let expected = [128.0 / 255.0, 64.0 / 255.0, 191.0 / 255.0];
        let untagged = encode_rgb_png([128, 64, 191], None);
        let malformed = encode_rgb_png([128, 64, 191], Some(b"not an ICC profile".to_vec()));

        assert_pixel_close("untagged PNG", decoded_pixel(&untagged), expected, 1.0e-6);
        assert_pixel_close("malformed PNG", decoded_pixel(&malformed), expected, 1.0e-6);
    }

    #[test]
    fn configured_psd_fixture_decodes_as_one_source_image() {
        let Some(path) = std::env::var_os("RAW_EDITOR_PSD_FIXTURE") else {
            return;
        };
        let path = std::path::PathBuf::from(path);
        let bytes = std::fs::read(&path).expect("read PSD fixture");
        let image = load_psd_image(&bytes, None).expect("decode PSD fixture");
        assert!(image.width() > 0);
        assert!(image.height() > 0);
        assert!(matches!(
            image,
            DynamicImage::ImageRgb16(_) | DynamicImage::ImageRgba16(_)
        ));
    }
}
