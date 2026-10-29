//! Focus score, decision, alignment, selection, colour and tone tests for the
//! focus-stack compositor.

use super::*;

#[test]
fn focus_box_blur_preserves_a_constant_score_map() {
    let source = vec![3.5f32; 5 * 4];
    let blurred = box_blur_focus_map(&source, 5, 4, 2);

    assert!(blurred.iter().all(|value| (*value - 3.5).abs() < 1e-6));
}

#[test]
fn focus_score_prefers_native_edges_over_a_blurred_layer() {
    const WIDTH: u32 = 128;
    const HEIGHT: u32 = 128;
    let sharp = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
        // Use aperiodic, high-frequency structure so the metric cannot
        // win by matching a single periodic phase after blur.
        let cell = ((x / 3 + y / 5 + (x * 7 + y * 11) % 5) & 1) as f32;
        let value = 0.18 + cell * 0.62;
        Rgb([value, value * 0.92, value * 0.84])
    });
    let blurred = image::imageops::blur(&sharp, 2.2);
    let mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
    let sharp_score = focus_score_map(&sharp, &mask);
    let blurred_score = focus_score_map(&blurred, &mask);
    let interior = |scores: &[f32]| {
        let mut values = Vec::new();
        for y in 8..(HEIGHT - 8) as usize {
            let row = y * WIDTH as usize;
            values.extend_from_slice(&scores[row + 8..row + (WIDTH - 8) as usize]);
        }
        values.iter().copied().sum::<f32>() / values.len().max(1) as f32
    };
    let sharp_mean = interior(&sharp_score);
    let blurred_mean = interior(&blurred_score);
    assert!(
        sharp_mean > blurred_mean * 1.8,
        "focus metric did not separate sharp={sharp_mean:.5} from blurred={blurred_mean:.5}"
    );
}

#[test]
fn focus_analysis_keeps_source_sampling_density_on_long_mosaics() {
    let (analysis_width, analysis_height) = focus_analysis_dimensions(31_121, 11_034, 9_504);

    assert!(analysis_width >= 4_500);
    assert!(analysis_height >= 1_500);
    assert!(u64::from(analysis_width) * u64::from(analysis_height) <= FOCUS_ANALYSIS_MAX_PIXELS);
}

#[test]
fn focus_translation_recovers_a_non_grid_aligned_displacement() {
    const WIDTH: u32 = 800;
    const HEIGHT: u32 = 800;
    const LEFT: i32 = 21;
    const TOP: i32 = 17;
    const SHIFT_X: i32 = 19;
    const SHIFT_Y: i32 = 11;
    let pattern = |x: i32, y: i32| {
        let x = x as f32;
        let y = y as f32;
        // A deterministic aperiodic texture keeps the correspondence
        // unique instead of letting a periodic texture pass the margin at
        // another offset.
        let xi = x as i32;
        let yi = y as i32;
        let hash = ((xi.wrapping_mul(73_856_093)
            ^ yi.wrapping_mul(19_349_663)
            ^ (xi.wrapping_mul(yi)).wrapping_mul(83_492_791))
            & 255) as f32
            / 255.0;
        let wave = (x * 0.017 + y * 0.043).sin() * 0.12;
        (0.12 + hash * 0.72 + wave).clamp(0.03, 0.97)
    };
    let candidate_image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
        let value = pattern(x as i32, y as i32);
        Rgb([value, value * 0.91, value * 0.78])
    });
    let merged_image = Rgb32FImage::from_fn(WIDTH + 80, HEIGHT + 80, |x, y| {
        let source_x = x as i32 - LEFT - SHIFT_X;
        let source_y = y as i32 - TOP - SHIFT_Y;
        let value = pattern(source_x, source_y);
        Rgb([value, value * 0.91, value * 0.78])
    });
    let candidate = RenderedFocusLayer {
        image: candidate_image,
        mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        foreground_mask: GrayImage::new(WIDTH, HEIGHT),
        relaxed_foreground_mask: GrayImage::new(WIDTH, HEIGHT),
        left: LEFT as u32,
        top: TOP as u32,
    };
    let merged_mask = GrayImage::from_pixel(WIDTH + 80, HEIGHT + 80, image::Luma([255]));

    let result = estimate_focus_layer_translation(&candidate, &merged_image, &merged_mask)
        .expect("synthetic overlap should produce a translation");

    assert_eq!((result.0, result.1), (SHIFT_X, SHIFT_Y));
    assert!(result.3 >= 8, "insufficient patch consensus: {:?}", result);
}

#[test]
fn focus_translation_accepts_aligned_layers_without_forcing_motion() {
    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 640;
    let pattern = |x: i32, y: i32| {
        let v = ((x * 17 + y * 29).rem_euclid(97) as f32) / 97.0;
        0.15 + 0.7 * v
    };
    let image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
        let value = pattern(x as i32, y as i32);
        Rgb([value, value * 0.8, value * 0.65])
    });
    let candidate = RenderedFocusLayer {
        image: image.clone(),
        mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        foreground_mask: GrayImage::new(WIDTH, HEIGHT),
        relaxed_foreground_mask: GrayImage::new(WIDTH, HEIGHT),
        left: 0,
        top: 0,
    };
    let mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));

    let result = estimate_focus_layer_translation(&candidate, &image, &mask);

    assert!(
        result.is_none(),
        "already aligned layers should remain untouched: {result:?}"
    );
}

#[test]
fn focus_decision_prefers_the_locally_sharper_layer() {
    let base_focus = vec![0.9, 0.8, 0.1, 0.1];
    let candidate_focus = vec![0.1, 0.1, 0.8, 0.9];
    let base_mask = GrayImage::from_pixel(4, 1, image::Luma([255]));
    let candidate_mask = GrayImage::from_pixel(4, 1, image::Luma([255]));

    let decision = focus_decision_mask(&base_focus, &candidate_focus, &base_mask, &candidate_mask);

    assert_eq!(decision.as_raw(), &[0, 0, 255, 255]);
}

#[test]
fn focus_decision_extends_clear_structure_over_an_adjacent_halo() {
    let base_focus = vec![0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2];
    let candidate_focus = vec![0.1, 0.1, 0.1, 0.8, 0.8, 0.22, 0.2];
    let base_mask = GrayImage::from_pixel(7, 1, image::Luma([255]));
    let candidate_mask = GrayImage::from_pixel(7, 1, image::Luma([255]));

    let decision = focus_decision_mask(&base_focus, &candidate_focus, &base_mask, &candidate_mask);

    assert_eq!(decision.as_raw()[2], 0);
    assert_eq!(decision.as_raw()[4], 255);
    assert_eq!(decision.as_raw()[5], 255);
}

#[test]
fn focus_decision_protects_base_structure_symmetrically() {
    let base_focus = vec![0.1, 0.1, 0.8, 0.8, 0.22, 0.2, 0.2];
    let candidate_focus = vec![0.9, 0.9, 0.1, 0.1, 0.2, 0.2, 0.2];
    let base_mask = GrayImage::from_pixel(7, 1, image::Luma([255]));
    let candidate_mask = GrayImage::from_pixel(7, 1, image::Luma([255]));

    let decision = focus_decision_mask(&base_focus, &candidate_focus, &base_mask, &candidate_mask);

    assert_eq!(decision.as_raw()[0], 255);
    assert_eq!(decision.as_raw()[2], 0);
    assert_eq!(decision.as_raw()[4], 0);
}

#[test]
fn focus_canvas_switch_suppression_keeps_canvas_owner_but_allows_ink() {
    let mut decision = GrayImage::from_pixel(2, 1, image::Luma([255]));
    let base = Rgb32FImage::from_fn(2, 1, |x, _| {
        if x == 0 {
            Rgb([0.32, 0.28, 0.24])
        } else {
            Rgb([0.05, 0.05, 0.05])
        }
    });
    let candidate = Rgb32FImage::from_fn(2, 1, |x, _| {
        if x == 0 {
            Rgb([0.40, 0.36, 0.32])
        } else {
            Rgb([0.01, 0.01, 0.01])
        }
    });
    let mask = GrayImage::from_pixel(2, 1, image::Luma([255]));

    suppress_focus_canvas_switches(&mut decision, &base, &candidate, &mask, &mask);

    assert_eq!(decision.as_raw(), &[0, 255]);
}

#[test]
fn focus_seam_keeps_painted_detail_on_one_source() {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    let canvas = Rgb([0.32, 0.28, 0.24]);
    let candidate_canvas = Rgb([0.40, 0.36, 0.32]);
    let mut base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, canvas);
    let candidate_image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
        if x == 32 && (16..48).contains(&y) {
            Rgb([0.01, 0.01, 0.01])
        } else {
            candidate_canvas
        }
    });
    let candidate = RenderedFocusLayer {
        image: candidate_image,
        mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        foreground_mask: GrayImage::new(WIDTH, HEIGHT),
        relaxed_foreground_mask: GrayImage::new(WIDTH, HEIGHT),
        left: 0,
        top: 0,
    };
    let mut base_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
    let merged_foreground_mask = GrayImage::new(WIDTH, HEIGHT);
    let decision = GrayImage::from_fn(WIDTH, HEIGHT, |x, _| {
        image::Luma([u8::from(x >= WIDTH / 2) * 255])
    });

    blend_focus_seam_band(
        &mut base,
        &mut base_mask,
        &merged_foreground_mask,
        &candidate,
        &decision,
    );

    assert_eq!(base.get_pixel(32, 32).0, [0.01, 0.01, 0.01]);
}

#[test]
fn hard_focus_selection_copies_source_pixels_without_averaging() {
    let mut base =
        Rgb32FImage::from_raw(3, 1, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]).unwrap();
    let candidate =
        Rgb32FImage::from_raw(3, 1, vec![0.0, 0.0, 1.0, 0.2, 0.3, 0.9, 0.7, 0.1, 0.8]).unwrap();
    let mut base_mask = GrayImage::from_raw(3, 1, vec![255, 255, 0]).unwrap();
    let candidate_mask = GrayImage::from_pixel(3, 1, image::Luma([255]));
    let decision = GrayImage::from_raw(3, 1, vec![0, 255, 0]).unwrap();

    hard_select_focus_pixels(
        &mut base,
        &mut base_mask,
        &candidate,
        &candidate_mask,
        &decision,
    );

    assert_eq!(base.get_pixel(0, 0).0, [1.0, 0.0, 0.0]);
    assert_eq!(base.get_pixel(1, 0).0, [0.2, 0.3, 0.9]);
    assert_eq!(base.get_pixel(2, 0).0, [0.7, 0.1, 0.8]);
    assert_eq!(base_mask.as_raw(), &[255, 255, 255]);
}

#[test]
fn focus_color_correction_removes_a_measured_channel_gain() {
    let base = Rgb32FImage::from_pixel(40, 16, Rgb([0.4, 0.5, 0.6]));
    let mut candidate_image = Rgb32FImage::from_pixel(32, 16, Rgb([0.5, 0.625, 0.75]));
    let base_mask = GrayImage::from_pixel(40, 16, image::Luma([255]));
    let candidate = RenderedFocusLayer {
        image: candidate_image.clone(),
        mask: GrayImage::from_pixel(32, 16, image::Luma([255])),
        foreground_mask: GrayImage::new(32, 16),
        relaxed_foreground_mask: GrayImage::new(32, 16),
        left: 4,
        top: 0,
    };

    let correction = estimate_focus_color_correction(
        &base,
        &base_mask,
        &GrayImage::new(40, 16),
        &candidate,
        false,
    );
    assert!(
        correction
            .gains
            .iter()
            .all(|gain| (*gain - 0.8).abs() < 0.01)
    );

    apply_focus_color_correction(
        &mut candidate_image,
        &correction,
        &GrayImage::new(32, 16),
        false,
    );
    let corrected = candidate_image.get_pixel(0, 0);
    assert!((corrected[0] - 0.4).abs() < 0.01);
    assert!((corrected[1] - 0.5).abs() < 0.01);
    assert!((corrected[2] - 0.6).abs() < 0.01);
}

#[test]
fn focus_color_correction_uses_only_consensus_foreground_overlap() {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 32;
    let base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.4, 0.4, 0.4]));
    let candidate_image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |_, y| {
        if y < 24 {
            Rgb([0.5, 0.5, 0.5])
        } else {
            Rgb([0.4, 0.4, 0.4])
        }
    });
    let foreground_mask = GrayImage::from_fn(WIDTH, HEIGHT, |_, y| {
        image::Luma([if y < 24 { 255 } else { 0 }])
    });
    let candidate = RenderedFocusLayer {
        image: candidate_image,
        mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        foreground_mask: foreground_mask.clone(),
        relaxed_foreground_mask: foreground_mask.clone(),
        left: 0,
        top: 0,
    };

    let correction = estimate_focus_color_correction(
        &base,
        &GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        &foreground_mask,
        &candidate,
        false,
    );

    assert!(
        correction
            .gains
            .iter()
            .all(|gain| (*gain - 0.8).abs() < 0.01)
    );
}

#[test]
fn focus_color_correction_can_use_bright_consensus_edge_pixels() {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 32;
    let base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.94, 0.94, 0.94]));
    let candidate = RenderedFocusLayer {
        image: Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.98, 0.98, 0.98])),
        mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        foreground_mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        relaxed_foreground_mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        left: 0,
        top: 0,
    };

    let correction = estimate_focus_color_correction(
        &base,
        &GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        &GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        &candidate,
        false,
    );

    assert!(
        correction
            .gains
            .iter()
            .all(|gain| (*gain - (0.94 / 0.98)).abs() < 0.01)
    );
}

#[test]
fn focus_color_correction_reduces_a_broad_local_gain_step() {
    const WIDTH: u32 = 1_536;
    const HEIGHT: u32 = 32;
    let base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.4, 0.5, 0.6]));
    let candidate_image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, _| {
        if x < WIDTH / 2 {
            Rgb([0.5, 0.625, 0.75])
        } else {
            Rgb([0.4, 0.5, 0.6])
        }
    });
    let candidate = RenderedFocusLayer {
        image: candidate_image.clone(),
        mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
        foreground_mask: GrayImage::new(WIDTH, HEIGHT),
        relaxed_foreground_mask: GrayImage::new(WIDTH, HEIGHT),
        left: 0,
        top: 0,
    };
    let base_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
    let correction = estimate_focus_color_correction(
        &base,
        &base_mask,
        &GrayImage::new(WIDTH, HEIGHT),
        &candidate,
        false,
    );
    assert!(correction.spatial_gains.is_some());

    let mut corrected = candidate_image;
    apply_focus_color_correction(
        &mut corrected,
        &correction,
        &GrayImage::new(WIDTH, HEIGHT),
        false,
    );
    let input_step = 0.5f32 - 0.4;
    let corrected_step =
        (corrected.get_pixel(100, 10)[0] - corrected.get_pixel(1_300, 10)[0]).abs();
    assert!(corrected_step < input_step * 0.8);
}

#[test]
fn focus_background_tone_harmonization_reduces_step_without_touching_foreground() {
    const WIDTH: u32 = 192;
    const HEIGHT: u32 = 96;
    let mut image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
        if (80..112).contains(&x) && (32..64).contains(&y) {
            Rgb([0.65, 0.10, 0.08])
        } else if x < WIDTH / 2 {
            Rgb([0.32, 0.28, 0.24])
        } else {
            Rgb([0.40, 0.36, 0.32])
        }
    });
    let original_foreground_pixel = *image.get_pixel(96, 48);
    let image_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
    let foreground_mask = GrayImage::from_fn(WIDTH, HEIGHT, |x, y| {
        image::Luma([u8::from((80..112).contains(&x) && (32..64).contains(&y)) * 255])
    });

    let input_step = (image.get_pixel(20, 48)[0] - image.get_pixel(172, 48)[0]).abs();
    harmonize_focus_background_tone(&mut image, &image_mask, &foreground_mask);
    let output_step = (image.get_pixel(20, 48)[0] - image.get_pixel(172, 48)[0]).abs();

    assert!(output_step < input_step * 0.5);
    assert_eq!(*image.get_pixel(96, 48), original_foreground_pixel);
}

#[test]
fn focus_owner_tone_harmonization_matches_source_blocks_without_touching_detail() {
    const WIDTH: u32 = 192;
    const HEIGHT: u32 = 96;
    let mut image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
        if (80..112).contains(&x) && (32..64).contains(&y) {
            Rgb([0.65, 0.10, 0.08])
        } else if x < WIDTH / 2 {
            Rgb([0.32, 0.28, 0.24])
        } else {
            Rgb([0.40, 0.36, 0.32])
        }
    });
    let original_foreground_pixel = *image.get_pixel(96, 48);
    let image_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
    let foreground_mask = GrayImage::new(WIDTH, HEIGHT);
    let owner_map = GrayImage::from_fn(WIDTH, HEIGHT, |x, _| {
        image::Luma([if x < WIDTH / 2 { 1 } else { 2 }])
    });

    let input_step = (image.get_pixel(20, 48)[0] - image.get_pixel(172, 48)[0]).abs();
    harmonize_focus_background_tone_with_owners(
        &mut image,
        &image_mask,
        &foreground_mask,
        &owner_map,
    );
    let output_step = (image.get_pixel(20, 48)[0] - image.get_pixel(172, 48)[0]).abs();

    assert!(output_step < input_step * 0.5);
    assert_eq!(*image.get_pixel(96, 48), original_foreground_pixel);
}

#[test]
fn focus_canvas_margin_fill_keeps_the_full_canvas_and_extends_texture() {
    let mut image = Rgb32FImage::from_fn(8, 6, |x, y| {
        let value = (x as f32 * 0.07 + y as f32 * 0.11).fract();
        Rgb([value, value * 0.8, value * 0.6])
    });
    let mut mask = GrayImage::new(8, 6);
    for y in 1..5 {
        for x in 1..7 {
            mask.put_pixel(x, y, image::Luma([255]));
        }
    }

    let dimensions = image.dimensions();
    let (filled, remaining) = fill_focus_canvas_margins(&mut image, &mut mask);

    assert_eq!(image.dimensions(), dimensions);
    assert!(filled > 0);
    assert_eq!(remaining, 0);
    assert!(mask.as_raw().iter().all(|&value| value > 0));
    assert!(image.get_pixel(0, 0).0.iter().any(|&value| value != 0.0));
}
