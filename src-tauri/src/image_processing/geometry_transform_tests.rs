use super::*;

fn coordinate_gradient(width: u32, height: u32) -> DynamicImage {
    DynamicImage::ImageRgb32F(Rgb32FImage::from_fn(width, height, |x, y| {
        image::Rgb([
            x as f32 / (width - 1) as f32,
            y as f32 / (height - 1) as f32,
            ((x * 7 + y * 11) % 31) as f32 / 30.0,
        ])
    }))
}

#[test]
fn float_geometry_sources_are_borrowed_without_rgba_staging() {
    let rgb = coordinate_gradient(97, 73);
    let rgb_raw = rgb
        .as_rgb32f()
        .expect("fixture must use RGB32F storage")
        .as_raw();
    let rgb_source = GeometryPixelSource::new(&rgb);
    assert_eq!(rgb_source.conversion_bytes(), 0);
    let GeometrySampleData::Rgb32F(rgb_samples) = rgb_source.sample_data() else {
        panic!("RGB32F source must remain borrowed");
    };
    assert_eq!(rgb_samples.as_ptr(), rgb_raw.as_ptr());

    let rgba = DynamicImage::ImageRgba32F(rgb.to_rgba32f());
    let rgba_raw = rgba
        .as_rgba32f()
        .expect("fixture must use RGBA32F storage")
        .as_raw();
    let rgba_source = GeometryPixelSource::new(&rgba);
    assert_eq!(rgba_source.conversion_bytes(), 0);
    let GeometrySampleData::Rgba32F(rgba_samples) = rgba_source.sample_data() else {
        panic!("RGBA32F source must remain borrowed");
    };
    assert_eq!(rgba_samples.as_ptr(), rgba_raw.as_ptr());

    let rgb8 = DynamicImage::ImageRgb8(RgbImage::new(97, 73));
    let converted = GeometryPixelSource::new(&rgb8);
    assert_eq!(converted.conversion_bytes(), 97 * 73 * 4 * size_of::<f32>());
    assert!(matches!(
        converted.sample_data(),
        GeometrySampleData::Rgba32F(_)
    ));

    let borrowed = GeometryPixelSource::borrowed(&rgb8);
    assert_eq!(borrowed.conversion_bytes(), 0);
    assert!(matches!(
        borrowed.sample_data(),
        GeometrySampleData::Rgb8(_)
    ));
}

#[test]
fn borrowed_geometry_sampler_matches_materialized_pixels_for_all_storage_types() {
    const WIDTH: u32 = 31;
    const HEIGHT: u32 = 23;
    let images = vec![
        DynamicImage::ImageLuma8(image::GrayImage::from_fn(WIDTH, HEIGHT, |x, y| {
            image::Luma([x.wrapping_mul(17).wrapping_add(y * 13) as u8])
        })),
        DynamicImage::ImageLumaA8(image::GrayAlphaImage::from_fn(WIDTH, HEIGHT, |x, y| {
            image::LumaA([
                x.wrapping_mul(17).wrapping_add(y * 13) as u8,
                x.wrapping_mul(7).wrapping_add(y * 3) as u8,
            ])
        })),
        DynamicImage::ImageRgb8(RgbImage::from_fn(WIDTH, HEIGHT, |x, y| {
            image::Rgb([
                x.wrapping_mul(17).wrapping_add(y * 13) as u8,
                x.wrapping_mul(5).wrapping_add(y * 19) as u8,
                x.wrapping_mul(11).wrapping_add(y * 7) as u8,
            ])
        })),
        DynamicImage::ImageRgba8(image::RgbaImage::from_fn(WIDTH, HEIGHT, |x, y| {
            image::Rgba([
                x.wrapping_mul(17).wrapping_add(y * 13) as u8,
                x.wrapping_mul(5).wrapping_add(y * 19) as u8,
                x.wrapping_mul(11).wrapping_add(y * 7) as u8,
                x.wrapping_mul(3).wrapping_add(y * 23) as u8,
            ])
        })),
        DynamicImage::ImageLuma16(image::ImageBuffer::from_fn(WIDTH, HEIGHT, |x, y| {
            image::Luma([x.wrapping_mul(3_071).wrapping_add(y * 2_053) as u16])
        })),
        DynamicImage::ImageLumaA16(image::ImageBuffer::from_fn(WIDTH, HEIGHT, |x, y| {
            image::LumaA([
                x.wrapping_mul(3_071).wrapping_add(y * 2_053) as u16,
                x.wrapping_mul(1_013).wrapping_add(y * 997) as u16,
            ])
        })),
        DynamicImage::ImageRgb16(image::ImageBuffer::from_fn(WIDTH, HEIGHT, |x, y| {
            image::Rgb([
                x.wrapping_mul(3_071).wrapping_add(y * 2_053) as u16,
                x.wrapping_mul(1_009).wrapping_add(y * 4_093) as u16,
                x.wrapping_mul(2_039).wrapping_add(y * 1_021) as u16,
            ])
        })),
        DynamicImage::ImageRgba16(image::ImageBuffer::from_fn(WIDTH, HEIGHT, |x, y| {
            image::Rgba([
                x.wrapping_mul(3_071).wrapping_add(y * 2_053) as u16,
                x.wrapping_mul(1_009).wrapping_add(y * 4_093) as u16,
                x.wrapping_mul(2_039).wrapping_add(y * 1_021) as u16,
                x.wrapping_mul(509).wrapping_add(y * 4_099) as u16,
            ])
        })),
        coordinate_gradient(WIDTH, HEIGHT),
        DynamicImage::ImageRgba32F(image::Rgba32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
            image::Rgba([
                x as f32 / (WIDTH - 1) as f32,
                y as f32 / (HEIGHT - 1) as f32,
                ((x * 7 + y * 11) % 31) as f32 / 30.0,
                ((x * 13 + y * 5) % 29) as f32 / 28.0,
            ])
        })),
    ];
    let params = GeometryParams {
        distortion: 9.0,
        projection: 17.0,
        vertical: 11.0,
        horizontal: -8.0,
        rotate: 2.5,
        aspect: 3.0,
        scale: 98.0,
        x_offset: 1.0,
        y_offset: -1.5,
        lens_dist_k1: 0.01,
        lens_dist_k2: -0.002,
        tca_vr: 1.002,
        tca_vb: 0.998,
        vig_k1: -0.06,
        ..GeometryParams::default()
    };

    for source in images {
        let expected = warp_image_geometry(&source, params);
        let sampler = GeometryWarpSampler::new(&source, params);
        assert_eq!(sampler.dimensions(), expected.dimensions());
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                assert_eq!(
                    sampler.sample_rgba8(x, y),
                    Some(expected.get_pixel(x, y)),
                    "{:?} changed at ({x}, {y})",
                    source.color()
                );
            }
        }
    }
}

#[test]
fn borrowed_geometry_sampler_matches_materialized_pixels_at_60mp_row_width() {
    const WIDTH: u32 = 9_504;
    const HEIGHT: u32 = 19;
    let source = coordinate_gradient(WIDTH, HEIGHT);
    let params = GeometryParams {
        distortion: 9.0,
        projection: 17.0,
        vertical: 11.0,
        horizontal: -8.0,
        rotate: 2.5,
        aspect: 3.0,
        scale: 98.0,
        x_offset: 1.0,
        y_offset: -1.5,
        lens_dist_k1: 0.01,
        lens_dist_k2: -0.002,
        tca_vr: 1.002,
        tca_vb: 0.998,
        vig_k1: -0.06,
        ..GeometryParams::default()
    };
    let expected = warp_image_geometry(&source, params);
    let sampler = GeometryWarpSampler::new(&source, params);

    for y in [0, 1, 7, 11, HEIGHT - 1] {
        for x in [0, 1, 63, 64, 2_047, 2_048, 4_751, 7_999, WIDTH - 1] {
            assert_eq!(
                sampler.sample_rgba8(x, y),
                Some(expected.get_pixel(x, y)),
                "geometry sampler changed the materialized pixel at ({x}, {y})"
            );
        }
    }
}

#[test]
fn borrowed_rgb_geometry_matches_explicit_rgba_staging() {
    let rgb = coordinate_gradient(161, 121);
    let staged_rgba = DynamicImage::ImageRgba32F(rgb.to_rgba32f());
    let params = GeometryParams {
        distortion: 12.0,
        projection: 24.0,
        vertical: 13.0,
        horizontal: -9.0,
        rotate: 3.5,
        aspect: 4.0,
        scale: 97.0,
        x_offset: 1.5,
        y_offset: -2.0,
        lens_dist_k1: 0.012,
        lens_dist_k2: -0.003,
        tca_vr: 1.0015,
        tca_vb: 0.998,
        vig_k1: -0.08,
        ..GeometryParams::default()
    };

    let borrowed = warp_image_geometry(&rgb, params);
    let staged = warp_image_geometry(&staged_rgba, params);

    assert_eq!(
        borrowed
            .as_rgba32f()
            .expect("geometry output must be RGBA32F")
            .as_raw(),
        staged
            .as_rgba32f()
            .expect("geometry output must be RGBA32F")
            .as_raw(),
        "borrowing RGB32F must preserve the previous explicitly staged RGBA32F pixels"
    );
}

#[test]
fn cropped_geometry_rows_match_the_materialized_rgba16f_pixels() {
    let source = coordinate_gradient(161, 121);
    let params = GeometryParams {
        distortion: 12.0,
        projection: 24.0,
        vertical: 13.0,
        horizontal: -9.0,
        rotate: 3.5,
        aspect: 4.0,
        scale: 97.0,
        x_offset: 1.5,
        y_offset: -2.0,
        lens_dist_k1: 0.012,
        lens_dist_k2: -0.003,
        tca_vr: 1.0015,
        tca_vb: 0.998,
        vig_k1: -0.08,
        ..GeometryParams::default()
    };
    let crop = serde_json::json!({
        "unit": "px",
        "x": 17.4,
        "y": 11.4,
        "width": 101.2,
        "height": 73.2
    });
    let materialized = apply_crop(warp_image_geometry(&source, params), &crop).into_owned();
    let materialized = materialized
        .as_rgba32f()
        .expect("materialized geometry crop must remain RGBA32F");
    let rows =
        GeometryWarpRows::try_new_transformed_borrowed(&source, params, 0, false, false, &crop)
            .expect("RGB32F geometry source must support row streaming");

    assert_eq!(rows.dimensions(), materialized.dimensions());
    assert_eq!(rows.crop_offset(), (17.4, 11.4));
    assert_eq!(
        rows.materialize()
            .as_rgba32f()
            .expect("fallback must materialize RGBA32F")
            .as_raw(),
        materialized.as_raw()
    );
    let width = rows.dimensions().0 as usize;
    let mut streamed = vec![f16::ZERO; width * 4];
    for y in 0..rows.dimensions().1 {
        rows.write_rgba16f_row(y, &mut streamed)
            .expect("write cropped geometry row");
        let start = y as usize * width * 4;
        let expected = materialized.as_raw()[start..start + width * 4]
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        assert_eq!(streamed, expected, "streamed geometry row {y}");
    }
}

#[test]
fn orthogonal_geometry_bands_match_materialized_transform_crop_pixels() {
    let source = coordinate_gradient(37, 23);
    let params = GeometryParams {
        distortion: 9.0,
        projection: 17.0,
        vertical: 8.0,
        horizontal: -6.0,
        rotate: 2.5,
        aspect: 3.0,
        scale: 98.0,
        x_offset: 0.75,
        y_offset: -1.25,
        lens_dist_k1: 0.01,
        lens_dist_k2: -0.002,
        tca_vr: 1.001,
        tca_vb: 0.999,
        vig_k1: -0.06,
        ..GeometryParams::default()
    };

    for orientation_steps in 0_u8..4 {
        let (transformed_width, transformed_height) = if orientation_steps % 2 == 1 {
            (source.height(), source.width())
        } else {
            source.dimensions()
        };
        let crop = serde_json::json!({
            "unit": "px",
            "x": 2.0,
            "y": 1.0,
            "width": transformed_width - 4,
            "height": transformed_height - 3
        });

        for (flip_horizontal, flip_vertical) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let warped = warp_image_geometry(&source, params);
            let rotated = apply_coarse_rotation(warped, orientation_steps);
            let flipped = apply_flip(rotated, flip_horizontal, flip_vertical);
            let expected = apply_crop(flipped, &crop).into_owned();
            let expected = expected
                .as_rgba32f()
                .expect("orthogonal materialized output must remain RGBA32F");
            let rows = GeometryWarpRows::try_new_transformed_borrowed(
                &source,
                params,
                orientation_steps,
                flip_horizontal,
                flip_vertical,
                &crop,
            )
            .expect("orthogonal RGB32F geometry source must stream");

            assert_eq!(rows.dimensions(), expected.dimensions());
            assert_eq!(
                rows.materialize()
                    .as_rgba32f()
                    .expect("fallback must preserve RGBA32F")
                    .as_raw(),
                expected.as_raw()
            );
            assert_eq!(rows.checkpoint_bytes() > 0, orientation_steps % 2 == 1);

            let (width, height) = rows.dimensions();
            let row_samples = width as usize * 4;
            let row_stride_samples = row_samples + 8;
            let sentinel = f16::from_f32(-1.0);
            let mut streamed = vec![sentinel; row_stride_samples * height as usize];
            if orientation_steps == 1 && !flip_horizontal && !flip_vertical {
                assert!(
                    rows.write_rgba16f_band(0, 0, row_stride_samples, &mut [])
                        .is_ok()
                );
                assert!(
                    rows.write_rgba16f_band(height, 1, row_stride_samples, &mut streamed)
                        .is_err()
                );
                assert!(
                    rows.write_rgba16f_band(0, 1, row_samples - 1, &mut streamed)
                        .is_err()
                );
                assert!(
                    rows.write_rgba16f_band(
                        0,
                        1,
                        row_stride_samples,
                        &mut streamed[..row_stride_samples - 1],
                    )
                    .is_err()
                );
            }
            for band_y in (0..height).step_by(3) {
                let band_rows = (height - band_y).min(3);
                let start = band_y as usize * row_stride_samples;
                let end = start + band_rows as usize * row_stride_samples;
                rows.write_rgba16f_band(
                    band_y,
                    band_rows,
                    row_stride_samples,
                    &mut streamed[start..end],
                )
                .expect("write transformed geometry band");
            }

            for y in 0..height as usize {
                let expected_start = y * row_samples;
                let expected_row = expected.as_raw()[expected_start..expected_start + row_samples]
                    .iter()
                    .copied()
                    .map(f16::from_f32)
                    .collect::<Vec<_>>();
                let streamed_start = y * row_stride_samples;
                assert_eq!(
                    &streamed[streamed_start..streamed_start + row_samples],
                    expected_row,
                    "orientation={orientation_steps} flip_h={flip_horizontal} flip_v={flip_vertical} row={y}"
                );
                assert!(
                    streamed[streamed_start + row_samples..streamed_start + row_stride_samples]
                        .iter()
                        .all(|sample| *sample == sentinel)
                );
            }
        }
    }
}

#[test]
fn geometry_buffer_plan_removes_one_60mp_rgba32f_frame() {
    const WIDTH: u32 = 9_504;
    const HEIGHT: u32 = 6_336;
    const RGBA32F_BYTES: usize = 963_477_504;

    let borrowed = GeometryWarpBufferPlan::new(WIDTH, HEIGHT, 0).expect("60MP buffer plan");
    let staged =
        GeometryWarpBufferPlan::new(WIDTH, HEIGHT, RGBA32F_BYTES).expect("legacy 60MP buffer plan");

    assert_eq!(borrowed.output_rgba32f_bytes, RGBA32F_BYTES);
    assert_eq!(borrowed.transient_bytes(), RGBA32F_BYTES);
    assert_eq!(staged.transient_bytes(), RGBA32F_BYTES * 2);
    assert_eq!(
        staged.transient_bytes() - borrowed.transient_bytes(),
        RGBA32F_BYTES
    );

    let upload = crate::render_strategy::GpuInputUploadPlan::new(WIDTH, HEIGHT)
        .expect("60MP GPU input upload plan");
    assert_eq!(upload.max_staging_bytes, 4_866_048);
    assert_eq!(
        borrowed.output_rgba32f_bytes - upload.max_staging_bytes,
        958_611_456
    );
}

#[test]
#[ignore = "manual deterministic 60MP geometry memory benchmark"]
fn synthetic_60mp_geometry_source_harness() {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(9_504_u32);
    let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(6_336_u32);
    let mode =
        std::env::var("RAW_EDITOR_GEOMETRY_BENCH_MODE").unwrap_or_else(|_| "borrowed".to_string());
    assert!(matches!(mode.as_str(), "borrowed" | "staged"));

    let running = Arc::new(AtomicBool::new(true));
    let peak_rss = Arc::new(AtomicU64::new(0));
    let sampler_running = Arc::clone(&running);
    let sampler_peak = Arc::clone(&peak_rss);
    let sampler = std::thread::spawn(move || {
        let Ok(pid) = sysinfo::get_current_pid() else {
            return;
        };
        let mut system = sysinfo::System::new();
        while sampler_running.load(Ordering::Relaxed) {
            system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
            if let Some(process) = system.process(pid) {
                sampler_peak.fetch_max(process.memory(), Ordering::Relaxed);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    });

    let source = DynamicImage::ImageRgb32F(Rgb32FImage::from_fn(width, height, |x, y| {
        image::Rgb([
            (x % 1024) as f32 / 1023.0,
            (y % 1024) as f32 / 1023.0,
            (x.wrapping_mul(17).wrapping_add(y * 13) % 1024) as f32 / 1023.0,
        ])
    }));
    let started = Instant::now();
    let staged_source = (mode == "staged").then(|| DynamicImage::ImageRgba32F(source.to_rgba32f()));
    let input = staged_source.as_ref().unwrap_or(&source);
    let output = warp_image_geometry(
        input,
        GeometryParams {
            projection: 18.0,
            vertical: 11.0,
            horizontal: -7.0,
            rotate: 2.0,
            ..GeometryParams::default()
        },
    );
    let elapsed = started.elapsed();
    std::thread::sleep(Duration::from_millis(25));
    running.store(false, Ordering::Relaxed);
    sampler.join().expect("join geometry RSS sampler");

    let pixels = u64::from(width) * u64::from(height);
    let source_rgb32f_bytes = pixels * 3 * size_of::<f32>() as u64;
    let rgba32f_frame_bytes = pixels * 4 * size_of::<f32>() as u64;
    let expected_live_pixel_bytes = source_rgb32f_bytes
        + rgba32f_frame_bytes
        + if mode == "staged" {
            rgba32f_frame_bytes
        } else {
            0
        };
    let output_raw = output
        .as_rgba32f()
        .expect("geometry benchmark output must be RGBA32F")
        .as_raw();
    let sample_stride = (output_raw.len() / 4_096).max(1);
    let sample_hash = output_raw
        .iter()
        .step_by(sample_stride)
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, sample| {
            (hash ^ u64::from(sample.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
        });
    println!(
        "{{\"mode\":\"{}\",\"width\":{},\"height\":{},\"elapsedMs\":{},\"peakRssBytes\":{},\"sourceRgb32fBytes\":{},\"rgba32fFrameBytes\":{},\"expectedLivePixelBytes\":{},\"sampleHash\":\"{:016x}\"}}",
        mode,
        width,
        height,
        elapsed.as_millis(),
        peak_rss.load(Ordering::Relaxed),
        source_rgb32f_bytes,
        rgba32f_frame_bytes,
        expected_live_pixel_bytes,
        sample_hash,
    );

    assert_eq!(output.dimensions(), (width, height));
    std::hint::black_box(source);
    std::hint::black_box(staged_source);
    std::hint::black_box(output);
}

#[test]
#[ignore = "manual deterministic 60MP streamed geometry output benchmark"]
fn synthetic_60mp_geometry_output_harness() {
    run_geometry_output_benchmark(
        "geometry",
        "RAW_EDITOR_GEOMETRY_OUTPUT_BENCH_MODE",
        0,
        false,
        false,
    );
}

#[test]
#[ignore = "manual deterministic 60MP orthogonal streamed geometry output benchmark"]
fn synthetic_60mp_orthogonal_geometry_output_harness() {
    run_geometry_output_benchmark(
        "orthogonal",
        "RAW_EDITOR_ORTHOGONAL_GEOMETRY_OUTPUT_BENCH_MODE",
        1,
        true,
        false,
    );
}

fn run_geometry_output_benchmark(
    benchmark: &str,
    mode_environment: &str,
    orientation_steps: u8,
    flip_horizontal: bool,
    flip_vertical: bool,
) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(9_504_u32);
    let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(6_336_u32);
    let mode = std::env::var(mode_environment).unwrap_or_else(|_| "streamed".to_string());
    assert!(matches!(mode.as_str(), "materialized" | "streamed"));

    let source = DynamicImage::ImageRgb32F(Rgb32FImage::from_fn(width, height, |x, y| {
        image::Rgb([
            (x % 1024) as f32 / 1023.0,
            (y % 1024) as f32 / 1023.0,
            (x.wrapping_mul(17).wrapping_add(y * 13) % 1024) as f32 / 1023.0,
        ])
    }));
    let params = GeometryParams {
        projection: 18.0,
        vertical: 11.0,
        horizontal: -7.0,
        rotate: 2.0,
        ..GeometryParams::default()
    };
    let transform = GeometryOrthogonalTransform::new(
        width,
        height,
        orientation_steps,
        flip_horizontal,
        flip_vertical,
    );
    let (output_width, output_height) = transform.dimensions();
    let upload_plan = crate::render_strategy::GpuInputUploadPlan::new(output_width, output_height)
        .expect("geometry output upload plan");
    let padded_row_samples = upload_plan.padded_bytes_per_row as usize / size_of::<f16>();
    let max_staging_samples = upload_plan.max_staging_bytes / size_of::<f16>();
    let mut staging = vec![f16::ZERO; max_staging_samples];

    let pid = sysinfo::get_current_pid().expect("resolve geometry output benchmark process");
    let mut baseline_system = sysinfo::System::new();
    baseline_system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    let baseline_rss = baseline_system
        .process(pid)
        .expect("read geometry output benchmark baseline")
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
    let materialized = (mode == "materialized").then(|| {
        let warped = warp_image_geometry(&source, params);
        let rotated = apply_coarse_rotation(warped, orientation_steps);
        apply_flip(rotated, flip_horizontal, flip_vertical).into_owned()
    });
    let streamed = (mode == "streamed").then(|| {
        GeometryWarpRows::try_new_transformed_borrowed(
            &source,
            params,
            orientation_steps,
            flip_horizontal,
            flip_vertical,
            &Value::Null,
        )
        .expect("stream benchmark uses borrowed RGB32F")
    });
    let pixel_count = output_width as usize * output_height as usize;
    let sample_stride = (pixel_count / 4_096).max(1);
    let mut next_sample = 0_usize;
    let mut sample_hash = 0xcbf2_9ce4_8422_2325_u64;

    for band_start in
        (0..output_height).step_by(crate::render_strategy::GPU_INPUT_UPLOAD_BAND_ROWS as usize)
    {
        let band_height = (output_height - band_start).min(upload_plan.band_rows);
        let band_samples = padded_row_samples * band_height as usize;
        staging[..band_samples].fill(f16::ZERO);
        if let Some(output) = materialized.as_ref() {
            let source = output
                .as_rgba32f()
                .expect("materialized geometry output must be RGBA32F")
                .as_raw();
            staging[..band_samples]
                .par_chunks_exact_mut(padded_row_samples)
                .enumerate()
                .for_each(|(row, output)| {
                    let source_row = (band_start as usize + row) * output_width as usize * 4;
                    for (output, value) in output[..output_width as usize * 4]
                        .iter_mut()
                        .zip(&source[source_row..source_row + output_width as usize * 4])
                    {
                        *output = f16::from_f32(*value);
                    }
                });
        } else {
            let rows = streamed.as_ref().expect("streamed geometry rows");
            rows.write_rgba16f_band(
                band_start,
                band_height,
                padded_row_samples,
                &mut staging[..band_samples],
            )
            .expect("stream geometry band");
        }

        let band_first_pixel = band_start as usize * output_width as usize;
        let band_end_pixel = band_first_pixel + band_height as usize * output_width as usize;
        while next_sample < band_end_pixel {
            if next_sample >= band_first_pixel {
                let local = next_sample - band_first_pixel;
                let row = local / output_width as usize;
                let column = local % output_width as usize;
                let offset = row * padded_row_samples + column * 4;
                for channel in &staging[offset..offset + 4] {
                    sample_hash = (sample_hash ^ u64::from(channel.to_bits()))
                        .wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
            next_sample = next_sample.saturating_add(sample_stride);
        }
    }

    let elapsed = started.elapsed();
    std::thread::sleep(Duration::from_millis(20));
    running.store(false, Ordering::Relaxed);
    sampler.join().expect("join geometry output RSS sampler");

    let pixels = u64::from(width) * u64::from(height);
    let rgba32f_output_bytes = pixels * 4 * size_of::<f32>() as u64;
    let checkpoint_bytes = streamed
        .as_ref()
        .map_or(0, GeometryWarpRows::checkpoint_bytes);
    let band_scratch_bytes = streamed
        .as_ref()
        .map_or(0, |rows| rows.max_band_scratch_bytes(upload_plan.band_rows));
    let materialized_geometry_bytes = if transform.is_identity() {
        rgba32f_output_bytes
    } else {
        rgba32f_output_bytes.saturating_mul(2)
    };
    let logical_cpu_peak_bytes = if mode == "materialized" {
        materialized_geometry_bytes.saturating_add(upload_plan.max_staging_bytes as u64)
    } else {
        (upload_plan.max_staging_bytes as u64)
            .saturating_add(checkpoint_bytes as u64)
            .saturating_add(band_scratch_bytes as u64)
    };
    let peak_rss = peak_rss.load(Ordering::Relaxed);
    println!(
        "{{\"benchmark\":\"{}\",\"mode\":\"{}\",\"sourceWidth\":{},\"sourceHeight\":{},\"outputWidth\":{},\"outputHeight\":{},\"orientationSteps\":{},\"flipHorizontal\":{},\"flipVertical\":{},\"elapsedMs\":{},\"baselineRssBytes\":{},\"peakRssBytes\":{},\"peakDeltaBytes\":{},\"rgba32fOutputBytes\":{},\"maxUploadStagingBytes\":{},\"checkpointBytes\":{},\"maxBandScratchBytes\":{},\"logicalCpuPeakBytes\":{},\"sampleHash\":\"{:016x}\"}}",
        benchmark,
        mode,
        width,
        height,
        output_width,
        output_height,
        orientation_steps,
        flip_horizontal,
        flip_vertical,
        elapsed.as_millis(),
        baseline_rss,
        peak_rss,
        peak_rss.saturating_sub(baseline_rss),
        rgba32f_output_bytes,
        upload_plan.max_staging_bytes,
        checkpoint_bytes,
        band_scratch_bytes,
        logical_cpu_peak_bytes,
        sample_hash,
    );
    std::hint::black_box(&source);
    std::hint::black_box(&materialized);
    std::hint::black_box(&streamed);
    std::hint::black_box(&staging);
}

fn absolute_difference(left: &DynamicImage, right: &DynamicImage) -> f64 {
    let left = left.to_rgb32f();
    let right = right.to_rgb32f();
    left.as_raw()
        .iter()
        .zip(right.as_raw())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum()
}

#[test]
fn every_geometry_menu_slider_changes_native_pixels() {
    let source = coordinate_gradient(161, 121);
    let mut cases = Vec::new();

    let projection = GeometryParams {
        projection: 70.0,
        ..GeometryParams::default()
    };
    cases.push(("projection", projection));

    let vertical = GeometryParams {
        vertical: 45.0,
        ..GeometryParams::default()
    };
    cases.push(("vertical", vertical));

    let horizontal = GeometryParams {
        horizontal: -45.0,
        ..GeometryParams::default()
    };
    cases.push(("horizontal", horizontal));

    let rotate = GeometryParams {
        rotate: 12.0,
        ..GeometryParams::default()
    };
    cases.push(("rotate", rotate));

    let aspect = GeometryParams {
        aspect: 25.0,
        ..GeometryParams::default()
    };
    cases.push(("aspect", aspect));

    let scale = GeometryParams {
        scale: 82.0,
        ..GeometryParams::default()
    };
    cases.push(("scale", scale));

    let x_offset = GeometryParams {
        x_offset: 12.0,
        ..GeometryParams::default()
    };
    cases.push(("x offset", x_offset));

    let y_offset = GeometryParams {
        y_offset: -12.0,
        ..GeometryParams::default()
    };
    cases.push(("y offset", y_offset));

    for (name, params) in cases {
        let transformed = warp_image_geometry(&source, params);
        assert!(
            absolute_difference(&source, &transformed) > 100.0,
            "{name} unexpectedly produced an identity image"
        );
    }
}

#[test]
fn preview_sampling_and_inverse_coordinates_share_the_same_geometry() {
    let width = 181;
    let height = 137;
    let source = coordinate_gradient(width, height);
    let adjustments = serde_json::json!({
        "transformProjection": 42.0,
        "transformVertical": 28.0,
        "transformHorizontal": -19.0,
        "transformRotate": 4.5,
        "transformAspect": 8.0,
        "transformScale": 96.0,
        "transformXOffset": 2.0,
        "transformYOffset": -3.0,
        "sectionVisibility": { "geometry": true, "optics": true }
    });
    let transformed = apply_geometry_warp(&source, &adjustments).into_owned();
    let target_x = 91;
    let target_y = 68;
    let (source_x, source_y) = inverse_transform_point(
        target_x as f64,
        target_y as f64,
        width as f64,
        height as f64,
        &adjustments,
    );
    let pixel = transformed.to_rgb32f().get_pixel(target_x, target_y).0;

    assert!((pixel[0] as f64 - source_x / (width - 1) as f64).abs() < 0.015);
    assert!((pixel[1] as f64 - source_y / (height - 1) as f64).abs() < 0.015);
}

#[test]
fn projection_forward_and_inverse_round_trip() {
    let (cx, cy, half_diagonal) = (500.0, 400.0, 640.312_423_7);
    for projection in [-100.0, -35.0, 35.0, 100.0] {
        let projected = project_geometry_point(820.0, 145.0, cx, cy, half_diagonal, projection);
        let restored =
            unproject_geometry_point(projected.0, projected.1, cx, cy, half_diagonal, projection);
        assert!((restored.0 - 820.0).abs() < 1e-6);
        assert!((restored.1 - 145.0).abs() < 1e-6);
    }
}

#[test]
fn geometry_warp_preserves_transparent_blank_edges() {
    let source = coordinate_gradient(161, 121);
    let params = GeometryParams {
        rotate: 14.0,
        ..GeometryParams::default()
    };
    let transformed = warp_image_geometry(&source, params).to_rgba32f();

    assert_eq!(transformed.get_pixel(0, 0)[3], 0.0);
    assert!(transformed.get_pixel(80, 60)[3] > 0.99);
}

#[test]
fn lens_distortion_does_not_implicitly_zoom_away_blank_edges() {
    let source = coordinate_gradient(161, 121);
    let params = GeometryParams {
        distortion: 45.0,
        ..GeometryParams::default()
    };
    let transformed = warp_image_geometry(&source, params).to_rgba32f();

    assert_eq!(transformed.get_pixel(0, 0)[3], 0.0);
    assert!(transformed.get_pixel(80, 60)[3] > 0.99);
}
