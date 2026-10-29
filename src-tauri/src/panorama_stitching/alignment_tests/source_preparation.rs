//! Source preparation and run budgets: RAW display gain and exposure normalisation,
//! generated-file filtering, worker and memory budgets, render scaling and the cache key.

use super::*;

#[test]
fn shared_raw_display_gain_preserves_bright_scan_channels() {
    let mut highs = vec![0.3; 40];
    highs.extend([0.8, 0.95]);
    let gain = shared_stack_raw_display_gain(&highs).unwrap();
    assert_eq!(gain, 1.0);
    assert!(0.95 * gain < 1.0);
    let warm =
        DynamicImage::ImageRgb32F(Rgb32FImage::from_pixel(32, 32, image::Rgb([0.8, 0.4, 0.2])));
    assert_eq!(raw_display_channel_percentile(&warm), Some(0.8));
    let gain =
        shared_stack_raw_display_gain(&[raw_display_channel_percentile(&warm).unwrap()]).unwrap();
    assert!((gain - 1.125).abs() < 1e-6);
    assert!(shared_stack_raw_display_gain(&[f32::NAN, -1.0]).is_none());
}

#[test]
fn raw_physical_exposure_normalisation_uses_camera_settings() {
    let mut metadata = HashMap::from([
        ("ExposureTime".into(), "1/3 s".into()),
        ("FNumber".into(), "f/4".into()),
        ("ISOSpeedRatings".into(), "1000".into()),
    ]);
    let reference = stack_raw_exposure_value(&metadata).unwrap();
    assert!((reference - 1000.0 / 48.0).abs() < 1e-9);
    metadata.insert("ExposureTime".into(), "1/4 s".into());
    assert!((reference / stack_raw_exposure_value(&metadata).unwrap() - 4.0 / 3.0).abs() < 1e-9);
    metadata.insert("ExposureTime".into(), "0.625 s".into());
    metadata.insert("FNumber".into(), "f/5".into());
    assert_eq!(stack_raw_exposure_value(&metadata), Some(25.0));
    metadata.insert("FNumber".into(), "f/0".into());
    assert!(stack_raw_exposure_value(&metadata).is_none());
    metadata.remove("ExposureTime");
    assert!(stack_raw_exposure_value(&metadata).is_none());
}

#[test]
fn large_stack_alignment_budget_tightens_as_source_count_grows() {
    assert_eq!(scalable_alignment_budget(64), (2_400, 1_600));
    assert_eq!(scalable_alignment_budget(128), (1_800, 1_100));
    assert_eq!(scalable_alignment_budget(200), (1_536, 800));
}

#[test]
fn focus_stack_cache_key_includes_proxy_contract_and_input_budget() {
    let one = focus_stack_cache_pipeline_version(1);
    let two = focus_stack_cache_pipeline_version(2);
    assert!(one.contains("proxy_version=proxy-v1"));
    assert!(one.contains("resize_filter=triangle"));
    assert_ne!(one, two);
}

#[test]
fn mixed_focal_detection_distinguishes_lens_changes_from_rounding_noise() {
    assert!(focal_lengths_span_multiple_lenses([35.0, 85.0]));
    assert!(!focal_lengths_span_multiple_lenses([34.0, 35.0, 36.0]));
    assert!(!focal_lengths_span_multiple_lenses([f64::NAN, -1.0, 85.0]));
}

#[test]
fn scalable_preparation_workers_are_bounded_by_images_cpu_and_memory() {
    let abundant_memory = 32 * PREPARATION_RAM_PER_WORKER_BYTES;

    assert_eq!(
        bounded_preparation_worker_count(185, 10, abundant_memory),
        MAX_SCALABLE_PREPARATION_WORKERS
    );
    assert_eq!(bounded_preparation_worker_count(185, 4, abundant_memory), 4);
    assert_eq!(
        bounded_preparation_worker_count(185, 10, 2 * PREPARATION_RAM_PER_WORKER_BYTES),
        2
    );
    assert_eq!(bounded_preparation_worker_count(3, 10, abundant_memory), 3);
    assert_eq!(bounded_preparation_worker_count(185, 10, 0), 1);
}

#[test]
fn oversized_panorama_canvas_is_scaled_to_the_memory_budget() {
    assert_eq!(memory_safe_panorama_render_scale(12_000, 8_000), 1.0);
    let scale = memory_safe_panorama_render_scale(248_296, 9_843);
    let scaled_pixels = (248_296.0 * scale).ceil() as u64 * (9_843.0 * scale).ceil() as u64;

    assert!(scale < 0.32);
    assert!(scaled_pixels <= MAX_IN_MEMORY_PANORAMA_PIXELS + 100_000);
}

#[test]
fn render_scale_is_applied_after_the_global_image_transform() {
    let source = HashMap::from([(
        7,
        Matrix3::new(1.0, 0.0, 400.0, 0.0, 1.0, -80.0, 0.0, 0.0, 1.0),
    )]);
    let scaled = scaled_homographies(&source, 0.25);
    let transform = scaled.get(&7).expect("scaled transform should exist");

    assert_eq!(transform[(0, 0)], 0.25);
    assert_eq!(transform[(1, 1)], 0.25);
    assert_eq!(transform[(0, 2)], 100.0);
    assert_eq!(transform[(1, 2)], -20.0);
}

#[test]
fn scaled_render_sources_preserve_the_output_coordinate_system() {
    let source = HashMap::from([(
        7,
        Matrix3::new(1.0, 0.0, 400.0, 0.0, 1.0, -80.0, 0.0, 0.0, 1.0),
    )]);
    let scale = 0.25;
    let scaled = scaled_source_render_homographies(&source, scale);
    let source_point = nalgebra::Point3::new(800.0, 200.0, 1.0);
    let expected =
        Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0) * source[&7] * source_point;
    let actual =
        scaled[&7] * nalgebra::Point3::new(source_point.x * scale, source_point.y * scale, 1.0);

    assert!((actual.x - expected.x).abs() < 1e-9);
    assert!((actual.y - expected.y).abs() < 1e-9);
    assert!((actual.z - expected.z).abs() < 1e-9);

    let mut image = test_image(7, "large.jpg");
    image.width = 4_672;
    image.height = 7_008;
    let rendered = scaled_render_image_info(&image, scale);
    assert_eq!(rendered.dimensions(), (1_168, 1_752));
    assert!(rendered.alignment_image.is_empty());
    assert!(rendered.features.is_empty());
}

#[test]
fn parallel_area_resize_preserves_constant_rgb8_sources() {
    let source = RgbImage::from_pixel(7, 5, image::Rgb([51, 102, 204]));
    let resized = resize_rgb8_area_to_rgb32f(&source, 3, 2);

    assert_eq!(resized.dimensions(), (3, 2));
    for pixel in resized.pixels() {
        assert!((pixel[0] - 0.2).abs() < 1e-6);
        assert!((pixel[1] - 0.4).abs() < 1e-6);
        assert!((pixel[2] - 0.8).abs() < 1e-6);
    }
}

#[test]
fn generated_stack_outputs_and_macos_sidecars_are_not_reused_as_sources() {
    assert!(is_generated_stitch_output("/tmp/DSC08897_FocusStack1.jpg"));
    assert!(is_generated_stitch_output(
        "/tmp/scan_Panorama_20260905.jpg"
    ));
    assert!(is_generated_stitch_output("/tmp/scan_Pano.png"));
    assert!(!is_generated_stitch_output("/tmp/DSC08897.jpg"));
    assert!(!is_generated_stitch_output("/tmp/focusstack_reference.jpg"));
    assert!(is_auxiliary_stitch_file("/tmp/._DSC08897.jpg"));
    assert!(!is_auxiliary_stitch_file("/tmp/DSC08897.jpg"));
}
