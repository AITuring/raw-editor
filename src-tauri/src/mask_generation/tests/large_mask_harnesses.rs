//! Ignored 60 MP mask benchmarks.
//!
//! Run by `npm run synthetic-mask:bench`, `synthetic-mask-compose:bench`,
//! `synthetic-range-mask:bench`, `synthetic-range-source:bench` and
//! `synthetic-ai-mask:bench`; each prints its measurements.

use super::*;

#[test]
#[ignore = "manual deterministic 60MP range-mask overlap scratch benchmark"]
fn synthetic_60mp_range_mask_overlap_scratch_harness() {
    let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(9_504_u32);
    let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(6_336_u32);
    let mode =
        std::env::var("RAW_EDITOR_RANGE_MASK_BENCH_MODE").unwrap_or_else(|_| "tiled".to_string());
    assert!(matches!(mode.as_str(), "full" | "tiled"));

    let grow_feather = GrowFeatherPlan::new(2.0, 1.0, width, height);
    let halo = grow_feather.halo();
    let mut composition = GrayImage::from_pixel(width, height, Luma([17]));
    let sub_mask = SubMask {
        id: "synthetic-range-mask".to_string(),
        mask_type: "color".to_string(),
        visible: true,
        invert: false,
        opacity: 100.0,
        mode: SubMaskMode::Additive,
        parameters: Value::Null,
    };
    let pid = sysinfo::get_current_pid().expect("resolve benchmark process id");
    let mut baseline_system = sysinfo::System::new();
    baseline_system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    let baseline_rss = baseline_system
        .process(pid)
        .expect("read benchmark process after output allocation")
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

    let pixel = |x: u32, y: u32| Luma([x.wrapping_mul(17).wrapping_add(y.wrapping_mul(13)) as u8]);
    let started = Instant::now();
    if mode == "full" {
        let mut full_sub_mask = GrayImage::from_fn(width, height, pixel);
        grow_feather.apply(&mut full_sub_mask);
        composite_sub_mask_tile(
            &mut composition,
            &full_sub_mask,
            0,
            0,
            SubMaskMode::Additive,
        );
    } else {
        for tile_y in (0..height).step_by(GPU_TILE_SIZE as usize) {
            let tile_height = (height - tile_y).min(GPU_TILE_SIZE);
            for tile_x in (0..width).step_by(GPU_TILE_SIZE as usize) {
                let tile_width = (width - tile_x).min(GPU_TILE_SIZE);
                let expanded_x = tile_x.saturating_sub(halo);
                let expanded_y = tile_y.saturating_sub(halo);
                let expanded_right = tile_x
                    .saturating_add(tile_width)
                    .saturating_add(halo)
                    .min(width);
                let expanded_bottom = tile_y
                    .saturating_add(tile_height)
                    .saturating_add(halo)
                    .min(height);
                let mut tile = GrayImage::from_fn(
                    expanded_right - expanded_x,
                    expanded_bottom - expanded_y,
                    |x, y| pixel(x + expanded_x, y + expanded_y),
                );
                grow_feather.apply(&mut tile);
                composite_sub_mask_tile_region(
                    &mut composition,
                    &mut tile,
                    tile_x - expanded_x,
                    tile_y - expanded_y,
                    tile_width,
                    tile_height,
                    tile_x,
                    tile_y,
                    &sub_mask,
                );
            }
        }
    }
    let elapsed = started.elapsed();
    std::thread::sleep(Duration::from_millis(10));
    running.store(false, Ordering::Relaxed);
    sampler.join().expect("join range-mask RSS sampler");

    let sample_stride = (composition.as_raw().len() / 4_096).max(1);
    let sample_hash = composition
        .as_raw()
        .iter()
        .step_by(sample_stride)
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, sample| {
            (hash ^ u64::from(*sample)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    let peak_rss = peak_rss.load(Ordering::Relaxed);
    let expected_scratch_bytes = if mode == "full" {
        u64::from(width) * u64::from(height) * 3
    } else {
        let expanded_width = width.min(GPU_TILE_SIZE.saturating_add(halo * 2));
        let expanded_height = height.min(GPU_TILE_SIZE.saturating_add(halo * 2));
        u64::from(expanded_width) * u64::from(expanded_height) * 3
    };
    println!(
        "{{\"mode\":\"{}\",\"width\":{},\"height\":{},\"haloPixels\":{},\"elapsedMs\":{},\"baselineRssBytes\":{},\"peakRssBytes\":{},\"peakDeltaBytes\":{},\"expectedScratchBytes\":{},\"sampleHash\":\"{:016x}\"}}",
        mode,
        width,
        height,
        halo,
        elapsed.as_millis(),
        baseline_rss,
        peak_rss,
        peak_rss.saturating_sub(baseline_rss),
        expected_scratch_bytes,
        sample_hash,
    );
    std::hint::black_box(composition);
}

#[test]
#[ignore = "manual deterministic 60MP range-mask geometry-source benchmark"]
fn synthetic_60mp_range_mask_geometry_source_harness() {
    let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(9_504_u32);
    let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(6_336_u32);
    let mode = std::env::var("RAW_EDITOR_RANGE_SOURCE_BENCH_MODE")
        .unwrap_or_else(|_| "sampled".to_string());
    assert!(matches!(mode.as_str(), "full" | "sampled"));

    let source = DynamicImage::ImageRgb32F(image::Rgb32FImage::from_fn(width, height, |x, y| {
        image::Rgb([
            (x.wrapping_mul(17).wrapping_add(y * 13) & 1023) as f32 / 1023.0,
            (x.wrapping_mul(5).wrapping_add(y * 19) & 1023) as f32 / 1023.0,
            (x.wrapping_mul(11).wrapping_add(y * 7) & 1023) as f32 / 1023.0,
        ])
    }));
    let geometry_params = crate::image_processing::GeometryParams {
        distortion: 5.0,
        projection: 8.0,
        vertical: 6.0,
        horizontal: -4.0,
        rotate: 1.0,
        aspect: 2.0,
        scale: 99.0,
        x_offset: 0.5,
        y_offset: -0.5,
        lens_dist_k1: 0.004,
        tca_vr: 1.0005,
        tca_vb: 0.9995,
        vig_k1: -0.02,
        ..crate::image_processing::GeometryParams::default()
    };
    let mask_parameters = ParametricMaskParameters {
        target_x: f64::from(width / 2),
        target_y: f64::from(height / 2),
        tolerance: 28.0,
        grow: 0.0,
        feather: 0.0,
        rotation: 0.0,
        flip_horizontal: false,
        flip_vertical: false,
        orientation_steps: 0,
    };

    let pid = sysinfo::get_current_pid().expect("resolve benchmark process id");
    let mut baseline_system = sysinfo::System::new();
    baseline_system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    let baseline_rss = baseline_system
        .process(pid)
        .expect("read benchmark process after source allocation")
        .memory();
    let running = Arc::new(AtomicBool::new(true));
    let peak_rss = Arc::new(AtomicU64::new(baseline_rss));
    let sampler_running = Arc::clone(&running);
    let sampler_peak = Arc::clone(&peak_rss);
    let rss_sampler = std::thread::spawn(move || {
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
    let (mask, checkpoint_bytes, maximum_checkpoint_bytes) = if mode == "full" {
        let warped = crate::image_processing::warp_image_geometry(&source, geometry_params);
        let mask = generate_color_bitmap_from_parameters(
            &mask_parameters,
            width,
            height,
            1.0,
            (0.0, 0.0),
            (0, 0),
            &warped,
        )
        .expect("materialized range mask must rasterize");
        std::hint::black_box(warped);
        (mask, 0, 0)
    } else {
        let geometry_source = GeometryWarpSampler::new(&source, geometry_params);
        let mask = generate_color_bitmap_from_parameters(
            &mask_parameters,
            width,
            height,
            1.0,
            (0.0, 0.0),
            (0, 0),
            &geometry_source,
        )
        .expect("sampled range mask must rasterize");
        let checkpoint_bytes = geometry_source.checkpoint_bytes();
        let maximum_checkpoint_bytes = geometry_source.max_checkpoint_bytes();
        (mask, checkpoint_bytes, maximum_checkpoint_bytes)
    };
    let elapsed = started.elapsed();
    std::thread::sleep(Duration::from_millis(10));
    running.store(false, Ordering::Relaxed);
    rss_sampler.join().expect("join range-source RSS sampler");

    let sample_stride = (mask.as_raw().len() / 4_096).max(1);
    let sample_hash = mask
        .as_raw()
        .iter()
        .step_by(sample_stride)
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, sample| {
            (hash ^ u64::from(*sample)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    let peak_rss = peak_rss.load(Ordering::Relaxed);
    let materialized_source_bytes = u64::from(width) * u64::from(height) * 4 * 4;
    println!(
        "{{\"mode\":\"{}\",\"width\":{},\"height\":{},\"elapsedMs\":{},\"baselineRssBytes\":{},\"peakRssBytes\":{},\"peakDeltaBytes\":{},\"materializedSourceBytes\":{},\"checkpointBytes\":{},\"maximumCheckpointBytes\":{},\"sampleHash\":\"{:016x}\"}}",
        mode,
        width,
        height,
        elapsed.as_millis(),
        baseline_rss,
        peak_rss,
        peak_rss.saturating_sub(baseline_rss),
        materialized_source_bytes,
        checkpoint_bytes,
        maximum_checkpoint_bytes,
        sample_hash,
    );
    std::hint::black_box(mask);
    std::hint::black_box(source);
}

#[test]
#[ignore = "manual deterministic 60MP AI-mask overlap scratch benchmark"]
fn synthetic_60mp_ai_mask_overlap_scratch_harness() {
    let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(9_504_u32);
    let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(6_336_u32);
    let mode =
        std::env::var("RAW_EDITOR_AI_MASK_BENCH_MODE").unwrap_or_else(|_| "tiled".to_string());
    assert!(matches!(mode.as_str(), "full" | "tiled"));

    let source = GrayImage::from_fn(width, height, |x, y| {
        Luma([x.wrapping_mul(17).wrapping_add(y.wrapping_mul(13)) as u8])
    });
    let mut composition = GrayImage::from_pixel(width, height, Luma([17]));
    let transform = AiMaskTransform::from_options(None, None, None, None);
    let depth_selection = AiDepthSelection::new(18.0, 82.0, 10.0, 10.0, 1.0);
    let grow_feather = GrowFeatherPlan::new(2.0, 1.0, width, height);
    let halo = depth_selection
        .blur_radius()
        .saturating_add(grow_feather.halo());
    let sub_mask = SubMask {
        id: "synthetic-ai-mask".to_string(),
        mask_type: "ai-depth".to_string(),
        visible: true,
        invert: false,
        opacity: 100.0,
        mode: SubMaskMode::Additive,
        parameters: Value::Null,
    };
    let pid = sysinfo::get_current_pid().expect("resolve benchmark process id");
    let mut baseline_system = sysinfo::System::new();
    baseline_system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    let baseline_rss = baseline_system
        .process(pid)
        .expect("read benchmark process after AI source and output allocation")
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
    if mode == "full" {
        let mut full_sub_mask =
            transform.rasterize(&source, width, height, 1.0, (0.0, 0.0), (0, 0));
        depth_selection.apply_pointwise(&mut full_sub_mask);
        depth_selection.apply_blur(&mut full_sub_mask);
        grow_feather.apply(&mut full_sub_mask);
        composite_sub_mask_tile(
            &mut composition,
            &full_sub_mask,
            0,
            0,
            SubMaskMode::Additive,
        );
    } else {
        for tile_y in (0..height).step_by(GPU_TILE_SIZE as usize) {
            let tile_height = (height - tile_y).min(GPU_TILE_SIZE);
            for tile_x in (0..width).step_by(GPU_TILE_SIZE as usize) {
                let tile_width = (width - tile_x).min(GPU_TILE_SIZE);
                let expanded_x = tile_x.saturating_sub(halo);
                let expanded_y = tile_y.saturating_sub(halo);
                let expanded_right = tile_x
                    .saturating_add(tile_width)
                    .saturating_add(halo)
                    .min(width);
                let expanded_bottom = tile_y
                    .saturating_add(tile_height)
                    .saturating_add(halo)
                    .min(height);
                let mut tile = transform.rasterize(
                    &source,
                    expanded_right - expanded_x,
                    expanded_bottom - expanded_y,
                    1.0,
                    (0.0, 0.0),
                    (expanded_x, expanded_y),
                );
                depth_selection.apply_pointwise(&mut tile);
                depth_selection.apply_blur(&mut tile);
                grow_feather.apply(&mut tile);
                composite_sub_mask_tile_region(
                    &mut composition,
                    &mut tile,
                    tile_x - expanded_x,
                    tile_y - expanded_y,
                    tile_width,
                    tile_height,
                    tile_x,
                    tile_y,
                    &sub_mask,
                );
            }
        }
    }
    let elapsed = started.elapsed();
    std::thread::sleep(Duration::from_millis(10));
    running.store(false, Ordering::Relaxed);
    sampler.join().expect("join AI-mask RSS sampler");

    let sample_stride = (composition.as_raw().len() / 4_096).max(1);
    let sample_hash = composition
        .as_raw()
        .iter()
        .step_by(sample_stride)
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, sample| {
            (hash ^ u64::from(*sample)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    let pixel_bytes = u64::from(width) * u64::from(height);
    let peak_rss = peak_rss.load(Ordering::Relaxed);
    let expected_scratch_bytes = if mode == "full" {
        pixel_bytes * 3
    } else {
        let expanded_width = width.min(GPU_TILE_SIZE.saturating_add(halo * 2));
        let expanded_height = height.min(GPU_TILE_SIZE.saturating_add(halo * 2));
        u64::from(expanded_width) * u64::from(expanded_height) * 3
    };
    println!(
        "{{\"mode\":\"{}\",\"width\":{},\"height\":{},\"haloPixels\":{},\"elapsedMs\":{},\"baselineRssBytes\":{},\"peakRssBytes\":{},\"peakDeltaBytes\":{},\"sourceAndOutputBytes\":{},\"expectedScratchBytes\":{},\"sampleHash\":\"{:016x}\"}}",
        mode,
        width,
        height,
        halo,
        elapsed.as_millis(),
        baseline_rss,
        peak_rss,
        peak_rss.saturating_sub(baseline_rss),
        pixel_bytes * 2,
        expected_scratch_bytes,
        sample_hash,
    );
    std::hint::black_box(source);
    std::hint::black_box(composition);
}

#[test]
#[ignore = "manual deterministic 60MP mask composition scratch benchmark"]
fn synthetic_60mp_mask_composition_scratch_harness() {
    let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(9_504_u32);
    let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(6_336_u32);
    let mode = std::env::var("RAW_EDITOR_MASK_COMPOSITION_BENCH_MODE")
        .unwrap_or_else(|_| "tiled".to_string());
    assert!(matches!(mode.as_str(), "full" | "tiled"));

    let mut composition = GrayImage::from_pixel(width, height, Luma([17]));
    let pid = sysinfo::get_current_pid().expect("resolve benchmark process id");
    let mut baseline_system = sysinfo::System::new();
    baseline_system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    let baseline_rss = baseline_system
        .process(pid)
        .expect("read benchmark process after output allocation")
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
    if mode == "full" {
        let full_sub_mask = GrayImage::from_fn(width, height, |x, y| {
            Luma([x.wrapping_mul(17).wrapping_add(y * 13) as u8])
        });
        std::thread::sleep(Duration::from_millis(10));
        let mut output = Some(composition);
        composite_sub_mask(&mut output, full_sub_mask, SubMaskMode::Additive);
        composition = output.expect("full-frame composition output");
    } else {
        for tile_y in (0..height).step_by(GPU_TILE_SIZE as usize) {
            let tile_height = (height - tile_y).min(GPU_TILE_SIZE);
            for tile_x in (0..width).step_by(GPU_TILE_SIZE as usize) {
                let tile_width = (width - tile_x).min(GPU_TILE_SIZE);
                let tile = GrayImage::from_fn(tile_width, tile_height, |x, y| {
                    Luma([(x + tile_x)
                        .wrapping_mul(17)
                        .wrapping_add((y + tile_y) * 13) as u8])
                });
                if tile_x == 0 && tile_y == 0 {
                    std::thread::sleep(Duration::from_millis(10));
                }
                composite_sub_mask_tile(
                    &mut composition,
                    &tile,
                    tile_x,
                    tile_y,
                    SubMaskMode::Additive,
                );
            }
        }
    }
    let elapsed = started.elapsed();
    std::thread::sleep(Duration::from_millis(10));
    running.store(false, Ordering::Relaxed);
    sampler.join().expect("join mask composition RSS sampler");

    let sample_stride = (composition.as_raw().len() / 4_096).max(1);
    let sample_hash = composition
        .as_raw()
        .iter()
        .step_by(sample_stride)
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, sample| {
            (hash ^ u64::from(*sample)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    let peak_rss = peak_rss.load(Ordering::Relaxed);
    let expected_scratch_bytes = if mode == "full" {
        u64::from(width) * u64::from(height)
    } else {
        u64::from(width.min(GPU_TILE_SIZE)) * u64::from(height.min(GPU_TILE_SIZE))
    };
    println!(
        "{{\"mode\":\"{}\",\"width\":{},\"height\":{},\"elapsedMs\":{},\"baselineRssBytes\":{},\"peakRssBytes\":{},\"peakDeltaBytes\":{},\"expectedScratchBytes\":{},\"sampleHash\":\"{:016x}\"}}",
        mode,
        width,
        height,
        elapsed.as_millis(),
        baseline_rss,
        peak_rss,
        peak_rss.saturating_sub(baseline_rss),
        expected_scratch_bytes,
        sample_hash,
    );
    std::hint::black_box(composition);
}

#[test]
#[ignore = "manual deterministic 60MP mask cache ownership benchmark"]
fn synthetic_60mp_mask_cache_ownership_harness() {
    enum Fixture {
        Cloned {
            cache: GrayImage,
            caller: GrayImage,
        },
        Shared {
            cache: Arc<GrayImage>,
            caller: Arc<GrayImage>,
        },
    }

    let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(9_504_u32);
    let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(6_336_u32);
    let mode = std::env::var("RAW_EDITOR_MASK_BENCH_MODE").unwrap_or_else(|_| "shared".to_string());
    assert!(matches!(mode.as_str(), "shared" | "cloned"));

    let pid = sysinfo::get_current_pid().expect("resolve benchmark process id");
    let mut baseline_system = sysinfo::System::new();
    baseline_system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    let baseline_rss = baseline_system
        .process(pid)
        .expect("read benchmark process before mask allocation")
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
            std::thread::sleep(Duration::from_millis(10));
        }
    });

    let started = Instant::now();
    let source = GrayImage::from_fn(width, height, |x, y| {
        Luma([x.wrapping_mul(17).wrapping_add(y * 13) as u8])
    });
    let fixture = if mode == "cloned" {
        Fixture::Cloned {
            caller: source.clone(),
            cache: source,
        }
    } else {
        let cache = Arc::new(source);
        Fixture::Shared {
            caller: Arc::clone(&cache),
            cache,
        }
    };
    let elapsed = started.elapsed();
    std::thread::sleep(Duration::from_millis(25));
    running.store(false, Ordering::Relaxed);
    sampler.join().expect("join mask RSS sampler");

    let (cache_raw, caller_raw, shared_allocation) = match &fixture {
        Fixture::Cloned { cache, caller } => (cache.as_raw(), caller.as_raw(), false),
        Fixture::Shared { cache, caller } => {
            assert!(Arc::ptr_eq(cache, caller));
            (cache.as_raw(), caller.as_raw(), true)
        }
    };
    assert_eq!(cache_raw, caller_raw);
    let sample_stride = (cache_raw.len() / 4_096).max(1);
    let sample_hash = cache_raw
        .iter()
        .step_by(sample_stride)
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, sample| {
            (hash ^ u64::from(*sample)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    let mask_bytes = u64::from(width) * u64::from(height);
    let expected_live_pixel_bytes = mask_bytes * if shared_allocation { 1 } else { 2 };
    println!(
        "{{\"mode\":\"{}\",\"width\":{},\"height\":{},\"elapsedMs\":{},\"baselineRssBytes\":{},\"peakRssBytes\":{},\"peakDeltaBytes\":{},\"maskBytes\":{},\"expectedLivePixelBytes\":{},\"sampleHash\":\"{:016x}\"}}",
        mode,
        width,
        height,
        elapsed.as_millis(),
        baseline_rss,
        peak_rss.load(Ordering::Relaxed),
        peak_rss
            .load(Ordering::Relaxed)
            .saturating_sub(baseline_rss),
        mask_bytes,
        expected_live_pixel_bytes,
        sample_hash,
    );
    std::hint::black_box(fixture);
}
