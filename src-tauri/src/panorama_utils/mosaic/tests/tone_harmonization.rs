//! Low-frequency exposure harmonization, source-owner correction and consensus
//! tests.

use super::*;

#[test]
fn low_frequency_harmonization_softens_group_exposure_steps_without_blurring_detail() {
    let store = StreamingMosaicStore::new(256, 64).expect("tile store should initialize");
    let rgb = Rgb32FImage::from_fn(256, 64, |x, y| {
        let base = if x < 128 { 0.42 } else { 0.58 };
        let detail = if (x / 4 + y / 4) % 2 == 0 {
            0.03
        } else {
            -0.03
        };
        Rgb([base + detail, base + detail, base + detail])
    });
    let mask = GrayImage::from_pixel(256, 64, Luma([255]));
    let owner = GrayImage::from_fn(256, 64, |x, _| Luma([if x < 128 { 1 } else { 2 }]));
    store
        .save_tile(0, 0, &rgb, &mask)
        .expect("tile should be writable");
    store
        .save_owner_tile(0, 0, &owner)
        .expect("owner should be writable");
    let (gains, scale) = streaming_seam_harmonization(&store, (0, 0, 256, 64), &[])
        .expect("harmonization should build")
        .expect("two owners should create a seam field");
    let mut output = rgb.clone();
    apply_streaming_seam_harmonization(&mut output, &gains, scale);
    let band_mean = |image: &Rgb32FImage, start: u32| {
        (start..start + 8)
            .map(|x| luma(*image.get_pixel(x, 32)))
            .sum::<f64>()
            / 8.0
    };
    let before_step = (band_mean(&rgb, 120) - band_mean(&rgb, 128)).abs();
    let after_step = (band_mean(&output, 120) - band_mean(&output, 128)).abs();
    assert!(
        after_step < before_step * 0.72,
        "the exposure seam should soften: {before_step:.4} -> {after_step:.4}"
    );
    let before_detail = (luma(*rgb.get_pixel(20, 20)) - luma(*rgb.get_pixel(24, 20))).abs();
    let after_detail = (luma(*output.get_pixel(20, 20)) - luma(*output.get_pixel(24, 20))).abs();
    assert!(
        after_detail >= before_detail * 0.90,
        "multiplicative low-frequency correction must retain local detail"
    );
}

#[test]
fn source_owner_low_frequency_correction_reduces_frame_step_and_keeps_detail() {
    let store = StreamingMosaicStore::new(256, 64).expect("tile store should initialize");
    let rgb = Rgb32FImage::from_fn(256, 64, |x, y| {
        let base = if x < 128 { 0.40 } else { 0.60 };
        let detail = if (x / 3 + y / 3) % 2 == 0 {
            0.025
        } else {
            -0.025
        };
        Rgb([base + detail, base + detail * 0.9, base + detail * 0.8])
    });
    let mask = GrayImage::from_pixel(256, 64, Luma([255]));
    let owner = GrayImage::from_fn(256, 64, |x, _| Luma([if x < 128 { 1 } else { 2 }]));
    store
        .save_tile(0, 0, &rgb, &mask)
        .expect("tile should be writable");
    store
        .save_owner_tile(0, 0, &owner)
        .expect("owner should be writable");
    let mut output = rgb.clone();
    let before_step = (luma(*output.get_pixel(120, 32)) - luma(*output.get_pixel(136, 32))).abs();
    let before_detail = (luma(*output.get_pixel(30, 30)) - luma(*output.get_pixel(33, 30))).abs();
    let corrected =
        apply_streaming_source_owner_low_frequency_correction(&store, (0, 0, 256, 64), &mut output)
            .expect("owner correction should run");
    let after_step = (luma(*output.get_pixel(120, 32)) - luma(*output.get_pixel(136, 32))).abs();
    let after_detail = (luma(*output.get_pixel(30, 30)) - luma(*output.get_pixel(33, 30))).abs();
    assert!(
        corrected > 0,
        "owner fields should contribute correction cells"
    );
    assert!(
        after_step < before_step * 0.72,
        "source exposure step should shrink: {before_step:.4} -> {after_step:.4}"
    );
    assert!(
        after_detail >= before_detail * 0.88,
        "owner correction must not erase source detail: {before_detail:.4} -> {after_detail:.4}"
    );
}

#[test]
fn source_owner_correction_reaches_the_interior_of_a_large_source_block() {
    let (width, height) = (2048, 64);
    let store = StreamingMosaicStore::new(width, height).expect("tile store should initialize");
    let rgb = Rgb32FImage::from_fn(width, height, |x, y| {
        let base = if x < width / 2 { 0.32 } else { 0.58 };
        let detail = if (x / 5 + y / 5) % 2 == 0 {
            0.02
        } else {
            -0.02
        };
        Rgb([base + detail, base + detail * 0.95, base + detail * 0.9])
    });
    let mask = GrayImage::from_pixel(width, height, Luma([255]));
    let owner = GrayImage::from_fn(width, height, |x, _| {
        Luma([if x < width / 2 { 1 } else { 2 }])
    });
    for column in 0..2 {
        let left = column * 1024;
        let rgb_tile = image::imageops::crop_imm(&rgb, left, 0, 1024, height).to_image();
        let mask_tile = image::imageops::crop_imm(&mask, left, 0, 1024, height).to_image();
        let owner_tile = image::imageops::crop_imm(&owner, left, 0, 1024, height).to_image();
        store
            .save_tile(column, 0, &rgb_tile, &mask_tile)
            .expect("tile should be writable");
        store
            .save_owner_tile(column, 0, &owner_tile)
            .expect("owner should be writable");
    }
    let mut output = rgb.clone();
    let before = (luma(*output.get_pixel(256, 32)) - luma(*output.get_pixel(1792, 32))).abs();
    let detail_before = (luma(*output.get_pixel(500, 32)) - luma(*output.get_pixel(505, 32))).abs();
    apply_streaming_source_owner_low_frequency_correction(
        &store,
        (0, 0, width, height),
        &mut output,
    )
    .expect("owner correction should run");
    let after = (luma(*output.get_pixel(256, 32)) - luma(*output.get_pixel(1792, 32))).abs();
    let detail_after = (luma(*output.get_pixel(500, 32)) - luma(*output.get_pixel(505, 32))).abs();
    assert!(
        after < before * 0.72,
        "large source step should shrink: {before:.4} -> {after:.4}"
    );
    assert!(
        detail_after >= detail_before * 0.88,
        "large source correction must retain detail"
    );
}

#[test]
fn streaming_consensus_adds_supported_low_frequency_delta_without_erasing_detail() {
    let width = 96;
    let height = 64;
    let base = Rgb32FImage::from_fn(width, height, |x, y| {
        let detail = if (x / 3 + y / 3) % 2 == 0 {
            0.025
        } else {
            -0.025
        };
        Rgb([0.42 + detail, 0.39 + detail, 0.34 + detail])
    });
    let candidate = Rgb32FImage::from_fn(width, height, |x, y| {
        let detail = if (x / 3 + y / 3) % 2 == 0 {
            0.025
        } else {
            -0.025
        };
        Rgb([0.52 + detail, 0.49 + detail, 0.44 + detail])
    });
    let mask = GrayImage::from_pixel(width, height, Luma([255]));
    let tone = Field::new(width, height, 112.0);
    let mut consensus = StreamingLowFrequencyConsensus::new(width, height);
    consensus.observe_overlap(0, 0, 1.0, &base, &mask, &candidate, &mask, &tone);
    let mut output = base.clone();
    let before_detail = (luma(*output.get_pixel(30, 30)) - luma(*output.get_pixel(33, 30))).abs();
    consensus.apply(&mut output, (0, 0, width, height));
    let after_detail = (luma(*output.get_pixel(30, 30)) - luma(*output.get_pixel(33, 30))).abs();
    let before_mean = luma(*base.get_pixel(48, 32));
    let after_mean = luma(*output.get_pixel(48, 32));
    assert!(
        after_mean > before_mean + 0.06,
        "consensus correction was not applied"
    );
    assert!(
        after_detail >= before_detail * 0.85,
        "low-frequency correction erased source detail: {before_detail} -> {after_detail}"
    );
}

#[test]
fn streaming_consensus_does_not_reapply_the_raw_candidate_exposure() {
    let width = 96;
    let height = 64;
    let base = Rgb32FImage::from_pixel(width, height, Rgb([0.60, 0.60, 0.60]));
    let candidate = Rgb32FImage::from_pixel(width, height, Rgb([0.30, 0.30, 0.30]));
    let mask = GrayImage::from_pixel(width, height, Luma([255]));
    let mut tone = Field::new(width, height, 112.0);
    tone.values.fill([2.0f64.ln(); 3]);
    let mut consensus = StreamingLowFrequencyConsensus::new(width, height);
    consensus.observe_overlap(0, 0, 1.0, &base, &mask, &candidate, &mask, &tone);
    let mut output = base.clone();
    consensus.apply(&mut output, (0, 0, width, height));
    let mut residual = 0.0f64;
    for y in 0..height {
        for x in 0..width {
            residual =
                residual.max((luma(*output.get_pixel(x, y)) - luma(*base.get_pixel(x, y))).abs());
        }
    }
    assert!(
        residual < 0.01,
        "consensus reapplied raw exposure after tone correction: {residual:.4}"
    );
}

#[test]
fn streaming_consensus_propagates_an_overlap_correction_into_a_source_block() {
    let width = 256;
    let height = 64;
    let base = Rgb32FImage::from_fn(width, height, |x, _| {
        let value = if x < 128 { 0.40 } else { 0.60 };
        Rgb([value, value, value])
    });
    let candidate = Rgb32FImage::from_pixel(width, height, Rgb([0.60, 0.60, 0.60]));
    let overlap = GrayImage::from_fn(width, height, |x, _| {
        Luma([u8::from((112..=144).contains(&x)) * 255])
    });
    let candidate_mask = GrayImage::from_pixel(width, height, Luma([255]));
    let tone = Field::new(width, height, 112.0);
    let mut consensus = StreamingLowFrequencyConsensus::new(width, height);
    consensus.observe_overlap(
        0,
        0,
        1.0,
        &base,
        &overlap,
        &candidate,
        &candidate_mask,
        &tone,
    );
    let mut output = base.clone();
    consensus.apply(&mut output, (0, 0, width, height));
    assert!(
        luma(*output.get_pixel(32, 32)) > 0.53,
        "the overlap correction should reach the low-exposure source block"
    );
    assert!(
        (luma(*output.get_pixel(32, 32)) - luma(*output.get_pixel(224, 32))).abs() < 0.08,
        "the source exposure step should be reduced across the block"
    );
}

#[test]
fn frequency_separation_removes_station_blocks_but_keeps_detail_and_black_bounds() {
    let mut image = Rgb32FImage::from_fn(512, 128, |x, y| {
        if y < 8 {
            return Rgb([0.0; 3]);
        }
        let base = if x < 256 { 0.36 } else { 0.54 };
        let detail = if (x / 3 + y / 3) % 2 == 0 {
            0.035
        } else {
            -0.035
        };
        Rgb([base + detail, base + detail, base + detail])
    });
    let original = image.clone();
    smooth_streaming_low_frequency_illumination(&mut image);
    let mean = |source: &Rgb32FImage, start: u32| {
        (start..start + 12)
            .map(|x| luma(*source.get_pixel(x, 64)))
            .sum::<f64>()
            / 12.0
    };
    let before = (mean(&original, 238) - mean(&original, 262)).abs();
    let after = (mean(&image, 238) - mean(&image, 262)).abs();
    assert!(
        after < before * 0.82,
        "station step {before:.4} -> {after:.4}"
    );
    let before_detail =
        (luma(*original.get_pixel(80, 64)) - luma(*original.get_pixel(83, 64))).abs();
    let after_detail = (luma(*image.get_pixel(80, 64)) - luma(*image.get_pixel(83, 64))).abs();
    assert!(after_detail >= before_detail * 0.88);
    assert!((0..512).all(|x| image.get_pixel(x, 2).0 == [0.0; 3]));
}

#[test]
fn same_coordinate_group_relations_recover_exposure_offset() {
    let (width, height) = (320, 160);
    let base = Rgb32FImage::from_fn(width, height, |x, y| {
        let detail = if (x / 5 + y / 5) % 2 == 0 {
            0.025
        } else {
            -0.025
        };
        Rgb([0.42 + detail, 0.39 + detail, 0.34 + detail])
    });
    let expected = [0.12f64, 0.08, -0.03];
    let candidate = Rgb32FImage::from_fn(width, height, |x, y| {
        let source = base.get_pixel(x, y);
        Rgb(std::array::from_fn(|channel| {
            source[channel] * expected[channel].exp() as f32
        }))
    });
    let mask = GrayImage::from_pixel(width, height, Luma([255]));
    let owner = GrayImage::from_pixel(width, height, Luma([1]));
    let tone = Field::new(width, height, 112.0);
    let relations = streaming_group_tone_relations_from_analysis(
        &base, &mask, &owner, &candidate, &mask, &tone, 2,
    );
    assert_eq!(relations.len(), 1);
    assert_eq!(relations[0].0, (1, 2));
    for (actual, expected) in relations[0].1.into_iter().zip(expected) {
        assert!((actual - expected).abs() < 0.01, "{actual} vs {expected}");
    }
}

#[test]
#[ignore = "applies production low-frequency correction to a supplied real preview"]
fn real_preview_low_frequency_correction_from_env() {
    let input = std::env::var_os("RAW_EDITOR_MOSAIC_PREVIEW_INPUT")
        .expect("RAW_EDITOR_MOSAIC_PREVIEW_INPUT is required");
    let output = std::env::var_os("RAW_EDITOR_MOSAIC_PREVIEW_OUTPUT")
        .expect("RAW_EDITOR_MOSAIC_PREVIEW_OUTPUT is required");
    let mut image = image::open(input)
        .expect("real preview should open")
        .to_rgb32f();
    smooth_streaming_low_frequency_illumination(&mut image);
    image::DynamicImage::ImageRgb32F(image)
        .to_rgb8()
        .save(output)
        .expect("corrected preview should save");
}

#[test]
fn group_exposure_harmonization_cannot_draw_a_halo_around_dark_strokes() {
    let store = StreamingMosaicStore::new(256, 64).expect("tile store should initialize");
    let rgb = Rgb32FImage::from_fn(256, 64, |x, _| {
        let paper = if x < 128 { 0.46 } else { 0.56 };
        let value = if (58..=62).contains(&x) { 0.06 } else { paper };
        Rgb([value, value * 0.96, value * 0.90])
    });
    let mask = GrayImage::from_pixel(256, 64, Luma([255]));
    let owner = GrayImage::from_fn(256, 64, |x, _| Luma([if x < 128 { 1 } else { 2 }]));
    store
        .save_tile(0, 0, &rgb, &mask)
        .expect("tile should be writable");
    store
        .save_owner_tile(0, 0, &owner)
        .expect("owner should be writable");
    let (gains, _) = streaming_seam_harmonization(&store, (0, 0, 256, 64), &[])
        .expect("harmonization should build")
        .expect("two owners should create a gain field");
    let paper_gain = gains.get_pixel(30, 32)[0];
    for x in [52, 56, 58, 60, 62, 64, 68] {
        assert!(
            (gains.get_pixel(x, 32)[0] - paper_gain).abs() < 1e-5,
            "a dark stroke must not alter the exposure gain around itself"
        );
    }
}

#[test]
fn smooth_tone_correction_preserves_selected_detail_relative_contrast() {
    let p = Rgb([0.2, 0.3, 0.4]);
    let q = Rgb([0.4, 0.5, 0.6]);
    let a = adjusted(p, [0.1, -0.05, 0.02]);
    let b = adjusted(q, [0.1, -0.05, 0.02]);
    for c in 0..3 {
        assert!((b[c] / a[c] - q[c] / p[c]).abs() < 1e-6);
    }
}
