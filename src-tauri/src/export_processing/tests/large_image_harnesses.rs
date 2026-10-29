//! Ignored 60 MP export benchmarks.
//!
//! Run by `npm run synthetic-webp:bench` and `npm run synthetic-export:bench`;
//! they print timing and peak RSS as JSON.

use super::*;

#[test]
#[ignore = "manual deterministic large-image encoder benchmark"]
fn synthetic_60mp_webp_output_harness() {
    let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(9_504_u32);
    let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(6_336_u32);
    let mode = std::env::var("RAW_EDITOR_BENCH_WEBP_MODE").unwrap_or_else(|_| "file".to_string());
    assert!(width > 0 && height > 0);

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

    let image = DynamicImage::ImageRgba8(RgbaImage::from_fn(width, height, |x, y| {
        Rgba([
            x.wrapping_mul(17).wrapping_add(y * 3) as u8,
            x.wrapping_mul(5).wrapping_add(y * 13) as u8,
            x.wrapping_mul(11).wrapping_add(y * 7) as u8,
            255,
        ])
    }));
    let benchmark_directory = tempfile::tempdir().expect("temporary WebP benchmark directory");
    let output_path = benchmark_directory.path().join("benchmark.webp");
    let started = Instant::now();
    let mut retained_memory = None;
    let output_bytes = match mode.as_str() {
        "memory" => {
            let encoded = encode_image_to_bytes(&image, "webp", 90).expect("encode WebP in memory");
            let output_bytes = encoded.len() as u64;
            retained_memory = Some(encoded);
            output_bytes
        }
        "file" => {
            save_webp_with_bounded_output(&image, &output_path, 90, &AtomicBool::new(false))
                .expect("encode WebP to bounded files");
            fs::metadata(&output_path)
                .expect("benchmark WebP metadata")
                .len()
        }
        _ => panic!("unsupported WebP benchmark mode: {mode}"),
    };
    let elapsed = started.elapsed();
    running.store(false, Ordering::Relaxed);
    sampler.join().expect("join RSS sampler");

    println!(
        "{{\"format\":\"webp\",\"mode\":\"{}\",\"width\":{},\"height\":{},\"elapsedMs\":{},\"outputBytes\":{},\"peakRssBytes\":{},\"inputRgbaBytes\":{}}}",
        mode,
        width,
        height,
        elapsed.as_millis(),
        output_bytes,
        peak_rss.load(Ordering::Relaxed),
        u64::from(width) * u64::from(height) * 4,
    );

    std::hint::black_box(retained_memory);
    std::hint::black_box(benchmark_directory);
}

#[test]
#[ignore = "manual deterministic large-image encoder benchmark"]
fn synthetic_60mp_streaming_export_harness() {
    use crate::render_strategy::StreamingExportBufferPlan;

    let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(9_504_u32);
    let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(6_336_u32);
    let output_format =
        std::env::var("RAW_EDITOR_BENCH_FORMAT").unwrap_or_else(|_| "png".to_string());
    let resize_long_edge = std::env::var("RAW_EDITOR_BENCH_RESIZE_LONG_EDGE")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0);
    let resize_options = resize_long_edge.map(|value| ResizeOptions {
        mode: ResizeMode::LongEdge,
        value,
        dont_enlarge: true,
    });
    let (target_width, target_height) = resize_options
        .as_ref()
        .map(|resize| calculate_resize_target(width, height, resize))
        .unwrap_or((width, height));
    let watermark_enabled = std::env::var("RAW_EDITOR_BENCH_WATERMARK")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "yes"));
    let metadata_enabled = std::env::var("RAW_EDITOR_BENCH_METADATA")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "yes"))
        && matches!(
            output_format.to_ascii_lowercase().as_str(),
            "jpg" | "jpeg" | "png" | "tif" | "tiff"
        );
    let benchmark_metadata = metadata_enabled.then(|| {
        let mut metadata = Metadata::new();
        metadata.set_tag(ExifTag::Make("Synthetic benchmark".to_string()));
        metadata.set_tag(ExifTag::Model("Bounded EXIF".to_string()));
        metadata.set_tag(ExifTag::Software("RAW Editor".to_string()));
        metadata
    });
    let is_tiff_output = matches!(output_format.to_ascii_lowercase().as_str(), "tif" | "tiff");
    let export_exif = benchmark_metadata
        .as_ref()
        .filter(|_| !is_tiff_output)
        .map(|metadata| {
            let encoded = metadata
                .as_u8_vec(FileExtension::JPEG)
                .expect("encode benchmark EXIF");
            assert_eq!(&encoded[..2], b"\xff\xe1");
            assert_eq!(&encoded[4..10], b"Exif\0\0");
            encoded[10..].to_vec()
        });
    let watermark_directory = watermark_enabled
        .then(|| tempfile::tempdir().expect("temporary benchmark watermark directory"));
    let watermark_settings = watermark_directory.as_ref().map(|directory| {
        let path = directory.path().join("benchmark-watermark.png");
        RgbaImage::from_fn(512, 128, |x, y| {
            Rgba([
                x.wrapping_mul(29).wrapping_add(y * 3) as u8,
                x.wrapping_mul(7).wrapping_add(y * 17) as u8,
                x.wrapping_mul(13).wrapping_add(y * 11) as u8,
                (96 + (x + y) % 160) as u8,
            ])
        })
        .save(&path)
        .expect("write benchmark watermark");
        WatermarkSettings {
            path: path.to_string_lossy().into_owned(),
            anchor: WatermarkAnchor::BottomRight,
            scale: 12.0,
            spacing: 5.0,
            opacity: 75.0,
        }
    });
    assert!(width > 0 && height > 0);

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

    let mut output = tempfile::tempfile().expect("temporary benchmark output");
    let buffer_plan = StreamingExportBufferPlan::new(width, height);
    let row_bytes = width as usize * 4;
    let mut band = vec![0_u8; buffer_plan.band_rgba_bytes()];
    let started = Instant::now();
    encode_streaming_rgba_rows(
        &mut output,
        target_width,
        target_height,
        &output_format,
        92,
        default_export_bit_depth_for_format(&output_format),
        true,
        if let Some(metadata) = benchmark_metadata.as_ref().filter(|_| is_tiff_output) {
            StreamingExportMetadata::TiffDirectories(metadata)
        } else if let Some(payload) = export_exif.as_deref() {
            StreamingExportMetadata::ExifTiffPayload(payload)
        } else {
            StreamingExportMetadata::None
        },
        |encoder_sink| {
            transform_streaming_rgba_rows(
                width,
                height,
                target_width,
                target_height,
                watermark_settings.as_ref(),
                encoder_sink,
                |source_sink| {
                    for band_y in (0..height).step_by(buffer_plan.band_rows as usize) {
                        let rows = buffer_plan.band_rows.min(height - band_y);
                        let active_band = &mut band[..rows as usize * row_bytes];
                        for (local_y, row) in active_band.chunks_exact_mut(row_bytes).enumerate() {
                            let y = band_y + local_y as u32;
                            for (x, pixel) in row.chunks_exact_mut(4).enumerate() {
                                let x = x as u32;
                                pixel[0] = x.wrapping_mul(17).wrapping_add(y * 3) as u8;
                                pixel[1] = x.wrapping_mul(5).wrapping_add(y * 13) as u8;
                                pixel[2] = x.wrapping_mul(11).wrapping_add(y * 7) as u8;
                                pixel[3] = 255;
                            }
                        }
                        for row in active_band.chunks_exact(row_bytes) {
                            source_sink(row)?;
                        }
                    }
                    Ok(())
                },
            )
        },
    )
    .expect("run synthetic streaming export");
    let elapsed = started.elapsed();
    running.store(false, Ordering::Relaxed);
    sampler.join().expect("join RSS sampler");
    let output_bytes = output.metadata().expect("benchmark output metadata").len();

    println!(
        "{{\"format\":\"{}\",\"width\":{},\"height\":{},\"targetWidth\":{},\"targetHeight\":{},\"watermark\":{},\"metadata\":{},\"elapsedMs\":{},\"outputBytes\":{},\"peakRssBytes\":{},\"producerBandBytes\":{},\"legacyFullFrameBytes\":{}}}",
        output_format,
        width,
        height,
        target_width,
        target_height,
        watermark_enabled,
        metadata_enabled,
        elapsed.as_millis(),
        output_bytes,
        peak_rss.load(Ordering::Relaxed),
        band.len(),
        buffer_plan.legacy_full_rgba_bytes()
    );
}
