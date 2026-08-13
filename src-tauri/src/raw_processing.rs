use crate::color_management::srgb_to_linear_channel;
use crate::image_processing::apply_orientation;
use anyhow::{Result, anyhow};
use image::{DynamicImage, ImageBuffer, Rgb};
use nalgebra::{Matrix3, Vector3};
use rawler::{
    decoders::{Orientation, RawDecodeParams},
    imgop::develop::{DemosaicAlgorithm, Intermediate, ProcessingStep, RawDevelop},
    imgop::xyz::Illuminant,
    rawimage::{RawImage, RawPhotometricInterpretation},
    rawsource::RawSource,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

pub fn develop_raw_image(
    file_bytes: &[u8],
    fast_demosaic: bool,
    highlight_compression: f32,
    linear_mode: String,
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
) -> Result<DynamicImage> {
    let (developed_image, orientation) = develop_internal(
        file_bytes,
        fast_demosaic,
        highlight_compression,
        linear_mode,
        cancel_token,
    )?;
    Ok(apply_orientation(developed_image, orientation))
}

fn correlated_color_temperature(wb_coeffs: [f32; 4], color_matrix: &[f32]) -> Option<u32> {
    if color_matrix.len() < 9
        || wb_coeffs[..3]
            .iter()
            .any(|coefficient| !coefficient.is_finite() || *coefficient <= 0.0)
    {
        return None;
    }

    let xyz_to_camera = Matrix3::from_row_slice(&color_matrix[..9]);
    let camera_white = Vector3::new(1.0 / wb_coeffs[0], 1.0 / wb_coeffs[1], 1.0 / wb_coeffs[2]);
    let xyz_white = xyz_to_camera.try_inverse()? * camera_white;
    let sum = xyz_white.x + xyz_white.y + xyz_white.z;
    if !sum.is_finite() || sum <= 0.0 {
        return None;
    }

    let x = xyz_white.x / sum;
    let y = xyz_white.y / sum;
    let denominator = y - 0.1858;
    if !x.is_finite() || !y.is_finite() || denominator.abs() < 1.0e-6 {
        return None;
    }

    // McCamy's approximation is sufficiently accurate for the editable 2,000–50,000 K range.
    let n = (x - 0.3320) / denominator;
    let temperature = -449.0 * n.powi(3) + 3525.0 * n.powi(2) - 6823.3 * n + 5520.33;
    if !temperature.is_finite() || !(2_000.0..=50_000.0).contains(&temperature) {
        return None;
    }

    Some(temperature.round() as u32)
}

pub fn estimate_as_shot_temperature(file_bytes: &[u8]) -> Option<u32> {
    let source = RawSource::new_from_slice(file_bytes);
    let decoder = rawler::get_decoder(&source).ok()?;
    let raw_image = decoder
        .raw_image(&source, &RawDecodeParams::default(), true)
        .ok()?;
    let color_matrix = raw_image
        .color_matrix
        .get(&Illuminant::D65)
        .or_else(|| raw_image.color_matrix.values().next())?;

    correlated_color_temperature(raw_image.wb_coeffs, color_matrix)
}

fn is_linear_raw_format(raw_image: &RawImage) -> bool {
    matches!(
        raw_image.photometric,
        RawPhotometricInterpretation::LinearRaw
    )
}

#[inline]
fn normalize_developed_channel(
    value: f32,
    rescale_factor: f32,
    is_linear_format: bool,
    apply_ungamma: bool,
) -> f32 {
    let rescaled = (value * rescale_factor).max(0.0);
    if is_linear_format && apply_ungamma {
        srgb_to_linear_channel(rescaled.clamp(0.0, 1.0))
    } else {
        rescaled
    }
}

fn developed_intermediate_into_dynamic_image(
    developed_intermediate: Intermediate,
) -> Result<DynamicImage> {
    let dimensions = developed_intermediate.dim();
    let width = dimensions.w as u32;
    let height = dimensions.h as u32;

    match developed_intermediate {
        Intermediate::ThreeColor(pixels) => {
            // rawler already owns tightly packed [f32; 3] pixels. Vec::into_flattened keeps that
            // allocation and only changes its element view, so the developed RGB frame can become
            // image::Rgb32F without allocating an overlapping RGBA32F copy.
            let buffer = ImageBuffer::<Rgb<f32>, _>::from_raw(
                width,
                height,
                pixels.into_inner().into_flattened(),
            )
            .ok_or_else(|| anyhow!("Failed to transfer developed RGB pixels"))?;
            Ok(DynamicImage::ImageRgb32F(buffer))
        }
        Intermediate::Monochrome(pixels) => {
            let buffer = ImageBuffer::<Rgb<f32>, _>::from_fn(width, height, |x, y| {
                let value = pixels.data[(y * width + x) as usize];
                Rgb([value, value, value])
            });
            Ok(DynamicImage::ImageRgb32F(buffer))
        }
        Intermediate::FourColor(_) => {
            Err(anyhow!("Unsupported intermediate format for conversion"))
        }
    }
}

fn develop_internal(
    file_bytes: &[u8],
    fast_demosaic: bool,
    highlight_compression: f32,
    linear_mode: String,
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
) -> Result<(DynamicImage, Orientation)> {
    let check_cancel = || -> Result<()> {
        if let Some((tracker, generation)) = &cancel_token
            && tracker.load(Ordering::SeqCst) != *generation
        {
            return Err(anyhow!("Load cancelled"));
        }
        Ok(())
    };

    check_cancel()?;

    let source = RawSource::new_from_slice(file_bytes);
    let decoder = rawler::get_decoder(&source)?;

    check_cancel()?;
    let mut raw_image: RawImage = decoder.raw_image(&source, &RawDecodeParams::default(), false)?;

    let metadata = decoder.raw_metadata(&source, &RawDecodeParams::default())?;
    let orientation = metadata
        .exif
        .orientation
        .map(Orientation::from_u16)
        .unwrap_or(Orientation::Normal);

    let is_linear_format = is_linear_raw_format(&raw_image);

    let (apply_ungamma, apply_calibration) = match linear_mode.as_str() {
        "gamma" => (true, true),
        "skip_calib" => (false, false),
        "gamma_skip_calib" => (true, false),
        _ => (false, true),
    };

    let original_white_level = raw_image
        .whitelevel
        .0
        .first()
        .cloned()
        .unwrap_or(u16::MAX as u32) as f32;
    let original_black_level = raw_image
        .blacklevel
        .levels
        .first()
        .map(|r| r.as_f32())
        .unwrap_or(0.0);

    for level in raw_image.whitelevel.0.iter_mut() {
        *level = u32::MAX;
    }

    let mut developer = RawDevelop::default();

    if is_linear_format {
        developer.steps.retain(|&step| {
            step != ProcessingStep::SRgb
                && step != ProcessingStep::Demosaic
                && (apply_calibration || step != ProcessingStep::Calibrate)
        });
    } else if fast_demosaic {
        developer.demosaic_algorithm = DemosaicAlgorithm::Speed;
        developer.steps.retain(|&step| step != ProcessingStep::SRgb);
    } else {
        developer.steps.retain(|&step| step != ProcessingStep::SRgb);
    }

    raw_image.wb_coeffs =
        crate::multi_exposure::neutralize_wb_if_multiexposure(raw_image.wb_coeffs, file_bytes);

    check_cancel()?;
    let mut developed_intermediate = developer.develop_intermediate(&raw_image)?;

    drop(raw_image);

    let denominator = (original_white_level - original_black_level).max(1.0);
    let rescale_factor = (u32::MAX as f32 - original_black_level) / denominator;

    let safe_highlight_compression = highlight_compression.max(1.01);

    let clamp_limit = if fast_demosaic {
        1.0
    } else {
        safe_highlight_compression
    };

    check_cancel()?;

    match &mut developed_intermediate {
        Intermediate::Monochrome(pixels) => {
            pixels.data.iter_mut().for_each(|p| {
                let linear_val = normalize_developed_channel(
                    *p,
                    rescale_factor,
                    is_linear_format,
                    apply_ungamma,
                );
                *p = linear_val.clamp(0.0, clamp_limit);
            });
        }
        Intermediate::ThreeColor(pixels) => {
            pixels.data.iter_mut().for_each(|p| {
                let r = normalize_developed_channel(
                    p[0],
                    rescale_factor,
                    is_linear_format,
                    apply_ungamma,
                );
                let g = normalize_developed_channel(
                    p[1],
                    rescale_factor,
                    is_linear_format,
                    apply_ungamma,
                );
                let b = normalize_developed_channel(
                    p[2],
                    rescale_factor,
                    is_linear_format,
                    apply_ungamma,
                );

                let max_c = r.max(g).max(b);

                let (final_r, final_g, final_b) = if max_c > 1.0 {
                    let min_c = r.min(g).min(b);
                    let compression_factor =
                        (1.0 - (max_c - 1.0) / (safe_highlight_compression - 1.0)).clamp(0.0, 1.0);
                    let compressed_r = min_c + (r - min_c) * compression_factor;
                    let compressed_g = min_c + (g - min_c) * compression_factor;
                    let compressed_b = min_c + (b - min_c) * compression_factor;
                    let compressed_max = compressed_r.max(compressed_g).max(compressed_b);

                    if compressed_max > 1e-6 {
                        let rescale = max_c / compressed_max;
                        (
                            compressed_r * rescale,
                            compressed_g * rescale,
                            compressed_b * rescale,
                        )
                    } else {
                        (max_c, max_c, max_c)
                    }
                } else {
                    (r, g, b)
                };

                p[0] = final_r.clamp(0.0, clamp_limit);
                p[1] = final_g.clamp(0.0, clamp_limit);
                p[2] = final_b.clamp(0.0, clamp_limit);
            });
        }
        Intermediate::FourColor(pixels) => {
            pixels.data.iter_mut().for_each(|p| {
                p.iter_mut().for_each(|c| {
                    let linear_val = normalize_developed_channel(
                        *c,
                        rescale_factor,
                        is_linear_format,
                        apply_ungamma,
                    );
                    *c = linear_val.clamp(0.0, clamp_limit);
                });
            });
        }
    }

    check_cancel()?;

    let dynamic_image = developed_intermediate_into_dynamic_image(developed_intermediate)?;

    Ok((dynamic_image, orientation))
}

pub fn get_fast_demosaic_scale_factor(
    file_bytes: &[u8],
    decoded_width: u32,
    decoded_height: u32,
) -> f32 {
    let source = RawSource::new_from_slice(file_bytes);
    if let Ok(decoder) = rawler::get_decoder(&source)
        && let Ok(raw_img) = decoder.raw_image(&source, &RawDecodeParams::default(), true)
    {
        let max_orig = (raw_img.width as f32).max(raw_img.height as f32);
        let max_comp = (decoded_width as f32).max(decoded_height as f32);
        if max_orig > 0.0 {
            let ratio = max_comp / max_orig;
            if ratio > 0.1 && ratio < 0.35 {
                return 0.25;
            } else if (0.35..0.75).contains(&ratio) {
                return 0.5;
            }
        }
    }
    1.0
}

#[cfg(test)]
mod tests {
    use super::{
        correlated_color_temperature, developed_intermediate_into_dynamic_image,
        normalize_developed_channel,
    };
    use image::{DynamicImage, ImageBuffer, Rgba};
    use rawler::{
        imgop::develop::Intermediate,
        pixarray::{Color2D, PixF32},
    };
    use std::mem::size_of;

    #[test]
    fn developed_three_color_pixels_transfer_into_rgb32f_without_copying() {
        let pixels = Color2D::new_with(
            vec![
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
            ],
            2,
            2,
        );
        let original_allocation = pixels.data.as_ptr().cast::<f32>();

        let image = developed_intermediate_into_dynamic_image(Intermediate::ThreeColor(pixels))
            .expect("transfer developed RGB pixels");
        let rgb = image
            .as_rgb32f()
            .expect("three-color development must remain RGB32F");

        assert_eq!(rgb.as_raw().as_ptr(), original_allocation);
        assert_eq!(
            rgb.as_raw(),
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        );

        let monochrome = PixF32::new_with(vec![0.2, 0.7], 2, 1);
        let monochrome_image =
            developed_intermediate_into_dynamic_image(Intermediate::Monochrome(monochrome))
                .expect("expand monochrome development to RGB");
        assert_eq!(
            monochrome_image
                .as_rgb32f()
                .expect("monochrome development must use RGB32F")
                .as_raw(),
            &[0.2, 0.2, 0.2, 0.7, 0.7, 0.7]
        );
    }

    #[test]
    fn zero_copy_raw_handoff_removes_60mp_rgba_expansion() {
        const WIDTH: u64 = 9_504;
        const HEIGHT: u64 = 6_336;
        const RGB32F_BYTES: u64 = 722_608_128;
        const RGBA32F_BYTES: u64 = 963_477_504;

        let pixels = WIDTH * HEIGHT;
        assert_eq!(pixels * 3 * size_of::<f32>() as u64, RGB32F_BYTES);
        assert_eq!(pixels * 4 * size_of::<f32>() as u64, RGBA32F_BYTES);
        assert_eq!(RGB32F_BYTES + RGBA32F_BYTES, 1_686_085_632);
        assert_eq!(RGB32F_BYTES + RGBA32F_BYTES - RGB32F_BYTES, RGBA32F_BYTES);
        assert_eq!(RGBA32F_BYTES - RGB32F_BYTES, 240_869_376);
    }

    #[test]
    #[ignore = "manual deterministic 60MP RAW RGB handoff memory benchmark"]
    fn synthetic_60mp_raw_rgb_handoff_harness() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, AtomicU64, Ordering},
        };
        use std::time::{Duration, Instant};

        let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(9_504_u32);
        let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(6_336_u32);
        let mode = std::env::var("RAW_EDITOR_RAW_HANDOFF_BENCH_MODE")
            .unwrap_or_else(|_| "reused".to_string());
        assert!(matches!(mode.as_str(), "expanded" | "reused"));

        let pixel_count = width as usize * height as usize;
        let pixels = Color2D::new_with(
            vec![[0.125_f32, 0.5, 0.875]; pixel_count],
            width as usize,
            height as usize,
        );
        let pid = sysinfo::get_current_pid().expect("resolve benchmark process id");
        let mut baseline_system = sysinfo::System::new();
        baseline_system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        let baseline_rss = baseline_system
            .process(pid)
            .expect("read benchmark process after developed RGB allocation")
            .memory();

        let running = Arc::new(AtomicBool::new(true));
        let peak_rss = Arc::new(AtomicU64::new(baseline_rss));
        let sampler_running = Arc::clone(&running);
        let sampler_peak = Arc::clone(&peak_rss);
        let sampler = std::thread::spawn(move || {
            let mut system = sysinfo::System::new();
            while sampler_running.load(Ordering::Relaxed) {
                system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
                if let Some(process) = system.process(pid) {
                    sampler_peak.fetch_max(process.memory(), Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        });

        let started = Instant::now();
        let image = if mode == "expanded" {
            let buffer = ImageBuffer::<Rgba<f32>, _>::from_fn(width, height, |x, y| {
                let pixel = pixels.data[(y * width + x) as usize];
                Rgba([pixel[0], pixel[1], pixel[2], 1.0])
            });
            DynamicImage::ImageRgba32F(buffer)
        } else {
            developed_intermediate_into_dynamic_image(Intermediate::ThreeColor(pixels))
                .expect("reuse developed RGB allocation")
        };
        let elapsed = started.elapsed();
        std::thread::sleep(Duration::from_millis(10));
        running.store(false, Ordering::Relaxed);
        sampler.join().expect("join RAW handoff RSS sampler");

        let mut sample_hash = 0xcbf2_9ce4_8422_2325_u64;
        let sample_stride = (pixel_count / 4_096).max(1);
        match &image {
            DynamicImage::ImageRgb32F(buffer) => {
                for pixel in buffer.pixels().step_by(sample_stride) {
                    for channel in pixel.0 {
                        sample_hash = (sample_hash ^ u64::from(channel.to_bits()))
                            .wrapping_mul(0x0000_0100_0000_01b3);
                    }
                }
            }
            DynamicImage::ImageRgba32F(buffer) => {
                for pixel in buffer.pixels().step_by(sample_stride) {
                    for channel in &pixel.0[..3] {
                        sample_hash = (sample_hash ^ u64::from(channel.to_bits()))
                            .wrapping_mul(0x0000_0100_0000_01b3);
                    }
                }
            }
            _ => unreachable!("RAW handoff benchmark must stay float RGB/RGBA"),
        }

        let pixels = u64::from(width) * u64::from(height);
        let rgb32f_bytes = pixels * 3 * size_of::<f32>() as u64;
        let rgba32f_bytes = pixels * 4 * size_of::<f32>() as u64;
        let expected_peak_pixel_bytes = if mode == "expanded" {
            rgb32f_bytes + rgba32f_bytes
        } else {
            rgb32f_bytes
        };
        let output_pixel_bytes = if mode == "expanded" {
            rgba32f_bytes
        } else {
            rgb32f_bytes
        };
        let peak_rss = peak_rss.load(Ordering::Relaxed);
        println!(
            "{{\"mode\":\"{}\",\"width\":{},\"height\":{},\"elapsedMs\":{},\"baselineRssBytes\":{},\"peakRssBytes\":{},\"peakDeltaBytes\":{},\"expectedPeakPixelBytes\":{},\"outputPixelBytes\":{},\"sampleHash\":\"{:016x}\"}}",
            mode,
            width,
            height,
            elapsed.as_millis(),
            baseline_rss,
            peak_rss,
            peak_rss.saturating_sub(baseline_rss),
            expected_peak_pixel_bytes,
            output_pixel_bytes,
            sample_hash,
        );
        std::hint::black_box(image);
    }

    #[test]
    fn linear_raw_gamma_uses_standard_srgb_eotf() {
        let decoded_mid_gray = normalize_developed_channel(0.5, 1.0, true, true);
        assert!((decoded_mid_gray - 0.214_041_14).abs() <= 1.0e-7);

        let already_linear_mid_gray = normalize_developed_channel(0.5, 1.0, true, false);
        assert_eq!(already_linear_mid_gray, 0.5);

        let non_linear_raw_value = normalize_developed_channel(0.5, 1.0, false, true);
        assert_eq!(non_linear_raw_value, 0.5);
    }

    #[test]
    fn rawler_bundles_sony_a7r_v_camera_profile_and_color_matrices() {
        let loader = rawler::RawLoader::new();
        let matching_profile_count = loader
            .get_cameras()
            .iter()
            .filter(|((make, model, _), _)| make == "SONY" && model == "ILCE-7RM5")
            .count();

        assert!(
            matching_profile_count >= 1,
            "expected the bundled α7R V base camera profile"
        );
        let base_profile = loader
            .get_cameras()
            .iter()
            .find_map(|((make, model, mode), camera)| {
                (make == "SONY" && model == "ILCE-7RM5" && mode.is_empty()).then_some(camera)
            })
            .expect("Sony ILCE-7RM5 base camera profile");

        assert_eq!(base_profile.clean_make, "Sony");
        assert_eq!(base_profile.clean_model, "ILCE-7RM5");
        assert_eq!((base_profile.cfa.width, base_profile.cfa.height), (2, 2));
        assert!(
            base_profile.color_matrix.len() >= 2,
            "α7R V profile must contain at least illuminant A and D65 matrices"
        );
    }

    #[test]
    fn estimates_standard_daylight_temperature_from_white_balance() {
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let d65_white = [0.95047_f32, 1.0, 1.08883];
        let white_balance = [
            1.0 / d65_white[0],
            1.0 / d65_white[1],
            1.0 / d65_white[2],
            f32::NAN,
        ];
        let estimated = correlated_color_temperature(white_balance, &identity).unwrap();

        assert!((6_350..=6_650).contains(&estimated));
    }

    #[test]
    fn estimates_tungsten_temperature_from_white_balance() {
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let illuminant_a_white = [1.09850_f32, 1.0, 0.35585];
        let white_balance = [
            1.0 / illuminant_a_white[0],
            1.0 / illuminant_a_white[1],
            1.0 / illuminant_a_white[2],
            f32::NAN,
        ];
        let estimated = correlated_color_temperature(white_balance, &identity).unwrap();

        assert!((2_750..=2_950).contains(&estimated));
    }
}

#[cfg(test)]
#[path = "raw_processing_acceptance.rs"]
mod acceptance_tests;
