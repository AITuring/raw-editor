//! Overlap exposure, seam-band and pyramid blending tests for the progressive
//! panorama stitcher.

use super::*;

#[test]
fn overlap_exposure_ignores_coloured_artwork_samples() {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    let base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.60, 0.50, 0.40]));
    let candidate = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, _| {
        if x < WIDTH * 3 / 4 {
            // A saturated red seal must not determine the paper gain.
            Rgb([1.0, 0.02, 0.01])
        } else {
            Rgb([0.60, 0.50, 0.40])
        }
    });
    let mut info = geometry_test_image(0, None);
    info.width = WIDTH;
    info.height = HEIGHT;
    let panorama_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
    let compensation = estimate_overlap_exposure_compensation(ExposureOverlap {
        panorama: &base,
        panorama_mask: &panorama_mask,
        candidate: &info,
        candidate_image: &candidate,
        candidate_inverse: &Matrix3::identity(),
        projection: Projection::Planar,
        offset_x: 0.0,
        offset_y: 0.0,
    });

    assert!(
        (compensation.representative_gain - 1.0).abs() < 0.02,
        "artwork contaminated exposure estimate: gain={:.3}",
        compensation.representative_gain
    );
}

#[test]
fn overlap_exposure_recovers_bounded_rgb_channel_gain() {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    let base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.66, 0.52, 0.40]));
    let candidate = Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.60, 0.55, 0.44]));
    let mut info = geometry_test_image(0, None);
    info.width = WIDTH;
    info.height = HEIGHT;
    let panorama_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
    let compensation = estimate_overlap_exposure_compensation(ExposureOverlap {
        panorama: &base,
        panorama_mask: &panorama_mask,
        candidate: &info,
        candidate_image: &candidate,
        candidate_inverse: &Matrix3::identity(),
        projection: Projection::Planar,
        offset_x: 0.0,
        offset_y: 0.0,
    });
    let gains = compensation.representative_channel_gain;
    assert!((gains[0] - 1.10).abs() < 0.02, "red gain={:.3}", gains[0]);
    assert!((gains[1] - 0.95).abs() < 0.02, "green gain={:.3}", gains[1]);
    assert!((gains[2] - 0.91).abs() < 0.02, "blue gain={:.3}", gains[2]);
}

#[test]
fn progressive_exposure_damping_keeps_direction_without_chain_drift() {
    let compensation = ExposureCompensation {
        cell_size: 1,
        grid_width: 1,
        grid_height: 1,
        gains: vec![0.75],
        representative_gain: 0.75,
        channel_gains: vec![[0.75, 1.20, 1.0]],
        representative_channel_gain: [0.75, 1.20, 1.0],
        use_spatial_field: false,
    }
    .damped(0.45);
    let expected = 0.75f32.ln().mul_add(0.45, 0.0).exp();
    assert!((compensation.representative_gain - expected).abs() < 1e-6);
    assert!(compensation.representative_gain < 1.0);
    assert!(compensation.representative_gain > 0.85);
    assert!(compensation.representative_channel_gain[1] > 1.0);
    assert_eq!(compensation.gain_at(0, 0), compensation.representative_gain);
}

#[test]
fn seam_blend_rejects_a_triangular_zero_support_corner() {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    let paper = Rgb([0.62, 0.51, 0.43]);
    let mut panorama = Rgb32FImage::from_pixel(WIDTH, HEIGHT, paper);
    let mut panorama_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
    let candidate = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
        if x + y < 28 {
            Rgb([0.0, 0.0, 0.0])
        } else {
            paper
        }
    });
    let mut info = geometry_test_image(0, None);
    info.width = WIDTH;
    info.height = HEIGHT;
    let exposure = ExposureCompensation {
        cell_size: 1,
        grid_width: 1,
        grid_height: 1,
        gains: vec![1.0],
        representative_gain: 1.0,
        channel_gains: vec![[1.0; 3]],
        representative_channel_gain: [1.0; 3],
        use_spatial_field: false,
    };
    let seam_coords = vec![HEIGHT as i32 / 2; WIDTH as usize];
    blend_panorama_seam_band(SeamBandBlend {
        panorama: &mut panorama,
        panorama_mask: &mut panorama_mask,
        img_to_add_info: &info,
        img_to_add: &candidate,
        h_add: &Matrix3::identity(),
        projection: Projection::Planar,
        offset_x: 0.0,
        offset_y: 0.0,
        orientation: SeamOrientation::Horizontal,
        seam_coords: &seam_coords,
        new_image_is_dominant_side: true,
        min_x: 0,
        max_x: WIDTH - 1,
        min_y: 0,
        max_y: HEIGHT - 1,
        exposure: &exposure,
    });

    let maximum_error = panorama
        .as_raw()
        .iter()
        .zip(std::iter::repeat(&paper.0).flatten())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        maximum_error < 1e-4,
        "zero-support corner leaked into the seam pyramid: error={maximum_error}"
    );
    assert!(!source_sample_has_support(&candidate, 4.0, 4.0));
    assert!(source_sample_has_support(&candidate, 40.0, 40.0));
}

#[test]
fn parallel_pyramid_downsample_preserves_constant_pixels_and_odd_edges() {
    let source = Rgb32FImage::from_pixel(5, 7, Rgb([0.2, 0.4, 0.8]));
    let downsampled = downsample_rgb_half(&source);

    assert_eq!(downsampled.dimensions(), (3, 4));
    assert!(downsampled.pixels().all(|pixel| pixel.0 == [0.2, 0.4, 0.8]));
}

#[test]
fn parallel_pyramid_detail_reconstructs_the_original_pixels() {
    let source = Rgb32FImage::from_fn(17, 11, |x, y| {
        let value = (x as f32 * 0.07 + y as f32 * 0.03).sin();
        Rgb([value, value * 0.5, 1.0 - value])
    });
    let coarse = downsample_rgb_half(&source);
    let detail = subtract_upsampled_rgb(&source, &coarse);
    let reconstructed = upsample_and_add_rgb(&coarse, &detail);

    assert_eq!(reconstructed.dimensions(), source.dimensions());
    let maximum_error = reconstructed
        .as_raw()
        .iter()
        .zip(source.as_raw())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(maximum_error < 1e-6, "maximum error: {maximum_error}");
}

#[test]
fn panorama_detail_feather_is_local_and_symmetric() {
    let radius = PANORAMA_DETAIL_SEAM_FEATHER_RADIUS;

    assert_eq!(panorama_detail_alpha(f32::NEG_INFINITY), 0.0);
    assert_eq!(panorama_detail_alpha(-radius), 0.0);
    assert_eq!(panorama_detail_alpha(0.0), 0.5);
    assert_eq!(panorama_detail_alpha(radius), 1.0);
    assert_eq!(panorama_detail_alpha(f32::INFINITY), 1.0);
}

#[test]
fn panorama_global_tone_blend_preserves_medium_frequency_detail() {
    const WIDTH: u32 = 256;
    const HEIGHT: u32 = 64;
    let pattern = |x: u32| if (x / 4).is_multiple_of(2) { 0.2 } else { 0.8 };
    let base = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, _| {
        let value = pattern(x);
        Rgb([value, value, value])
    });
    let candidate = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, _| {
        let value = (pattern((x + 4).min(WIDTH - 1)) + 0.08).min(1.0);
        Rgb([value, value, value])
    });
    let seam_mask = GrayImage::from_fn(WIDTH, HEIGHT, |x, _| {
        image::Luma([if x >= WIDTH / 2 { 255 } else { 0 }])
    });
    let tone_mask = GrayImage::from_fn(WIDTH, HEIGHT, |x, _| {
        image::Luma([((x as f32 / (WIDTH - 1) as f32) * 255.0).round() as u8])
    });
    let blended = multiband_blend(
        base.clone(),
        candidate,
        seam_mask,
        Some(tone_mask),
        PANORAMA_BLEND_BANDS,
        false,
    );
    let horizontal_variation = |image: &Rgb32FImage| {
        (32..96)
            .map(|x| {
                (image.get_pixel(x, HEIGHT / 2)[0] - image.get_pixel(x - 1, HEIGHT / 2)[0]).abs()
            })
            .sum::<f32>()
    };

    let source_variation = horizontal_variation(&base);
    let blended_variation = horizontal_variation(&blended);
    assert!(
        blended_variation >= source_variation * 0.9,
        "medium-frequency contrast fell from {source_variation:.3} to {blended_variation:.3}",
    );
}
