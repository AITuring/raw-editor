//! Source ownership tests: sharpness and sensor noise, disagreement, group
//! transitions and the in-memory focus mosaic.

use super::*;

#[test]
fn focus_group_ownership_cannot_reclaim_another_camera_position() {
    let sharp = sharp_fixture();
    let base = image::imageops::blur(&sharp, 2.0);
    let mask = GrayImage::from_pixel(sharp.width(), sharp.height(), Luma([255]));
    let owner = GrayImage::from_fn(sharp.width(), sharp.height(), |x, _| {
        Luma([if x < sharp.width() / 2 { 3 } else { 4 }])
    });
    let decision = streaming_ownership_from_analysis(
        &base,
        &mask,
        &sharp,
        &mask,
        &Field::new(sharp.width(), sharp.height(), 32.0),
        &owner,
        3,
        false,
        0.0,
        0.0,
        None,
        None,
    );
    let left_selected = decision
        .enumerate_pixels()
        .filter(|(x, _, pixel)| *x < decision.width() / 2 && pixel[0] != 0)
        .count();
    let right_selected = decision
        .enumerate_pixels()
        .filter(|(x, _, pixel)| *x >= decision.width() / 2 && pixel[0] != 0)
        .count();
    assert!(
        left_selected > 0,
        "the sharper member should contribute inside its group"
    );
    assert_eq!(
        right_selected, 0,
        "a focal member must not overwrite a different camera position"
    );
}

#[test]
fn panorama_group_transition_only_enters_from_new_coverage() {
    let candidate = sharp_fixture();
    let base = image::imageops::blur(&candidate, 1.0);
    let mut base_mask = GrayImage::new(candidate.width(), candidate.height());
    let mut owner = GrayImage::new(candidate.width(), candidate.height());
    for y in 0..candidate.height() {
        for x in 0..candidate.width() * 3 / 4 {
            base_mask.put_pixel(x, y, Luma([255]));
            owner.put_pixel(x, y, Luma([2]));
        }
    }
    let candidate_mask = GrayImage::from_pixel(candidate.width(), candidate.height(), Luma([255]));
    let decision = streaming_ownership_from_analysis(
        &base,
        &base_mask,
        &candidate,
        &candidate_mask,
        &Field::new(candidate.width(), candidate.height(), 32.0),
        &owner,
        3,
        true,
        0.0,
        0.0,
        None,
        None,
    );
    let left_quarter_selected = decision
        .enumerate_pixels()
        .filter(|(x, _, pixel)| *x < decision.width() / 4 && pixel[0] != 0)
        .count();
    let overlap_selected = decision
        .enumerate_pixels()
        .filter(|(x, _, pixel)| {
            *x >= decision.width() / 2 && *x < decision.width() * 3 / 4 && pixel[0] != 0
        })
        .count();
    let new_coverage_selected = decision
        .enumerate_pixels()
        .filter(|(x, _, pixel)| *x >= decision.width() * 3 / 4 && pixel[0] != 0)
        .count();
    assert_eq!(
        left_quarter_selected, 0,
        "a new camera group must not form a detached sharpness island"
    );
    assert!(
        new_coverage_selected > 0,
        "new source coverage must be admitted"
    );
    assert!(
        overlap_selected > 0,
        "a panorama seam must be allowed to leave the rectangular new-coverage boundary"
    );
}

#[test]
fn panorama_transition_marks_ink_cells_as_expensive_seam_routes() {
    let image = Rgb32FImage::from_fn(96, 64, |x, y| {
        let ink = (38..58).contains(&x) && (18..46).contains(&y);
        let value = if ink { 0.05 } else { 0.52 };
        Rgb([value, value * 0.96, value * 0.90])
    });
    let quiet = streaming_transition_structure(|x, y| rgb_at(&image, x, y), 0.0, 0.0, 24.0);
    let ink = streaming_transition_structure(|x, y| rgb_at(&image, x, y), 30.0, 12.0, 36.0);
    assert!(quiet < 0.05, "plain paper should remain a cheap seam route");
    assert!(
        ink > 0.75,
        "a cell crossed by dark calligraphy must strongly repel the panorama seam"
    );
}

/// Silk-like weave under broad dark strokes: the weave carries the focus
/// evidence, the strokes make a registration error visible at low frequency.
fn scan_fixture(shift: u32) -> Rgb32FImage {
    Rgb32FImage::from_fn(240, 120, |x, y| {
        let x = x + shift;
        let stroke = if (x / 14).is_multiple_of(3) && (y / 30).is_multiple_of(2) {
            -0.28
        } else {
            0.0
        };
        let weave = 0.035 * ((x as f32 * 2.4).sin() + (y as f32 * 2.2).sin());
        let value = 0.55 + stroke + weave;
        Rgb([value * 1.1, value, value * 0.8])
    })
}

fn transition_cells_taken(
    base: &Rgb32FImage,
    candidate: &Rgb32FImage,
    transition_focus_weight: f64,
) -> (usize, usize) {
    let (width, height) = candidate.dimensions();
    let base_mask = GrayImage::from_fn(width, height, |x, _| {
        Luma([if x < width * 3 / 4 { 255 } else { 0 }])
    });
    let owner = GrayImage::from_fn(width, height, |x, _| {
        Luma([if x < width * 3 / 4 { 2 } else { 0 }])
    });
    let candidate_mask = GrayImage::from_pixel(width, height, Luma([255]));
    let decision = streaming_ownership_from_analysis(
        base,
        &base_mask,
        candidate,
        &candidate_mask,
        &Field::new(width, height, 32.0),
        &owner,
        3,
        true,
        transition_focus_weight,
        0.0,
        None,
        None,
    );
    let overlap = |x: u32| x >= decision.width() / 8 && x < decision.width() * 5 / 8;
    let taken = decision
        .enumerate_pixels()
        .filter(|(x, _, pixel)| overlap(*x) && pixel[0] != 0)
        .count();
    let total = decision
        .enumerate_pixels()
        .filter(|(x, _, _)| overlap(*x))
        .count();
    (taken, total)
}

#[test]
fn focus_scan_transition_lets_the_sharper_aligned_position_own_the_overlap() {
    let sharp = scan_fixture(0);
    let blurred = image::imageops::blur(&sharp, 1.5);
    let (spatial, total) = transition_cells_taken(&blurred, &sharp, 0.0);
    let (focused, _) = transition_cells_taken(&blurred, &sharp, 1.0);
    // The spatial prior alone keeps the whole interior of the overlap; the
    // weave-covered silk makes every seam path expensive, so the sharper
    // position wins large connected parts rather than every cell.
    assert!(
        focused > total / 6 && focused > spatial + total / 10,
        "the sharper position must own a large part of a consistent overlap: {focused}/{total} vs spatial {spatial}"
    );
}

#[test]
fn focus_scan_transition_replaces_a_darker_blurred_frame_edge_band() {
    // The established frame's edge band is darker from lens falloff and soft;
    // a registered sharper position must still be able to replace it even
    // though the two disagree in absolute level.
    let sharp = scan_fixture(0);
    let soft = image::imageops::blur(&sharp, 1.5);
    let darker_soft = Rgb32FImage::from_fn(sharp.width(), sharp.height(), |x, y| {
        Rgb(soft.get_pixel(x, y).0.map(|value| value * 0.8))
    });
    let (spatial, total) = transition_cells_taken(&darker_soft, &sharp, 0.0);
    let (focused, _) = transition_cells_taken(&darker_soft, &sharp, 1.0);
    assert!(
        focused > total / 6 && focused > spatial + total / 10,
        "a darker soft band must not block the sharper registered position: {focused}/{total} vs spatial {spatial}"
    );
}

#[test]
fn focus_scan_transition_keeps_the_base_where_the_sharper_position_is_displaced() {
    let blurred = image::imageops::blur(&scan_fixture(0), 1.5);
    let (aligned, total) = transition_cells_taken(&blurred, &scan_fixture(0), 1.0);
    let (displaced, _) = transition_cells_taken(&blurred, &scan_fixture(7), 1.0);
    assert!(
        displaced * 2 < aligned,
        "a displaced sharp layer must not take the overlap: {displaced} vs aligned {aligned} of {total}"
    );
}

#[test]
fn focus_bracket_never_takes_covered_pixels_inside_its_own_edge_margin() {
    // A sharper bracket whose border rows carry a bright lens-correction
    // streak: the streak scores as detail, so without the margin the focus
    // contest would paste it into the painting.
    let sharp = scan_fixture(0);
    let (width, height) = sharp.dimensions();
    let base = image::imageops::blur(&sharp, 1.5);
    let candidate = Rgb32FImage::from_fn(width, height, |x, y| match y {
        0..=2 => Rgb([0.95, 0.95, 0.95]),
        3 => Rgb([0.05, 0.05, 0.05]),
        _ => *sharp.get_pixel(x, y),
    });
    let mask = GrayImage::from_pixel(width, height, Luma([255]));
    let owner = GrayImage::from_pixel(width, height, Luma([3]));
    let decision = |edge_margin_px| {
        streaming_ownership_from_analysis(
            &base,
            &mask,
            &candidate,
            &mask,
            &Field::new(width, height, 32.0),
            &owner,
            3,
            false,
            0.0,
            edge_margin_px,
            None,
            None,
        )
    };
    let guarded = decision(8.0);
    let in_margin = guarded
        .enumerate_pixels()
        .filter(|(x, y, pixel)| {
            let near_edge = *x < 8 || *y < 8 || *x >= width - 8 || *y >= height - 8;
            near_edge && pixel[0] != 0
        })
        .count();
    let interior = guarded
        .enumerate_pixels()
        .filter(|(_, _, pixel)| pixel[0] != 0)
        .count();
    assert_eq!(
        in_margin, 0,
        "no covered cell inside the edge margin may change owner"
    );
    assert!(
        interior > 0,
        "the sharper bracket must still win its interior"
    );
}

fn sharp_fixture() -> Rgb32FImage {
    Rgb32FImage::from_fn(160, 120, |x, y| {
        let value = 0.36
            + 0.12 * (x as f32 * 0.75).sin()
            + 0.10 * (y as f32 * 0.65).cos()
            + 0.03 * ((x + y) as f32 * 0.19).sin();
        Rgb([value * 1.1, value, value * 0.8])
    })
}

fn noisy_colour_stroke_pair() -> (Rgb32FImage, Rgb32FImage) {
    let sharp = Rgb32FImage::from_fn(192, 160, |x, y| {
        let stroke = (x as i32 - 96).abs() <= 3 && (20..140).contains(&y);
        // A faded red mark can have almost the same luminance as its
        // brown substrate; colour edges still carry genuine focus.
        if stroke {
            Rgb([0.46, 0.31, 0.27])
        } else {
            Rgb([0.40, 0.34, 0.27])
        }
    });
    let blurred = image::imageops::blur(&sharp, 3.2);
    let noisy = |im: &Rgb32FImage, amplitude: f32| {
        Rgb32FImage::from_fn(im.width(), im.height(), |x, y| {
            let hash = x.wrapping_mul(0x9e3779b9) ^ y.wrapping_mul(0x85ebca6b);
            let hash = (hash ^ (hash >> 16)).wrapping_mul(0x7feb352d);
            let noise = ((hash & 65535) as f32 / 65535.0 - 0.5) * amplitude;
            Rgb(im.get_pixel(x, y).0.map(|v| (v + noise).clamp(0.0, 1.0)))
        })
    };
    (noisy(&sharp, 0.020), noisy(&blurred, 0.080))
}

#[test]
fn sensor_noise_cannot_hide_a_sharper_faint_colour_stroke() {
    let (sharp, noisy_blur) = noisy_colour_stroke_pair();
    let info = image_info(1, &sharp);
    let sampler = LayerSampler {
        info: &info,
        source: &sharp,
        source_divisor: 1.0,
        inverse: Matrix3::identity(),
        native_mapping: None,
        projection: Projection::Planar,
        offset: (0.0, 0.0),
        left: 0,
        top: 0,
        scale: 1.0,
        residual: Field::new(192, 160, 48.0),
    };
    let mask = GrayImage::from_pixel(192, 160, Luma([255]));
    let chosen = ownership(
        &noisy_blur,
        &mask,
        &sampler,
        &Field::new(192, 160, 56.0),
        192,
        160,
    );
    let selected = (30..130)
        .filter(|&y| chosen.get_pixel(96, y)[0] > 0)
        .count();
    assert!(
        selected >= 90,
        "faint focused colour must beat noisy defocus ({selected}/100)"
    );
}

#[test]
fn extra_sensor_noise_cannot_reclaim_an_already_focused_stroke() {
    let (sharp, noisy_blur) = noisy_colour_stroke_pair();
    let info = image_info(1, &noisy_blur);
    let sampler = LayerSampler {
        info: &info,
        source: &noisy_blur,
        source_divisor: 1.0,
        inverse: Matrix3::identity(),
        native_mapping: None,
        projection: Projection::Planar,
        offset: (0.0, 0.0),
        left: 0,
        top: 0,
        scale: 1.0,
        residual: Field::new(192, 160, 48.0),
    };
    let mask = GrayImage::from_pixel(192, 160, Luma([255]));
    let chosen = ownership(
        &sharp,
        &mask,
        &sampler,
        &Field::new(192, 160, 56.0),
        192,
        160,
    );
    let selected = (30..130)
        .filter(|&y| chosen.get_pixel(96, y)[0] > 0)
        .count();
    assert_eq!(
        selected, 0,
        "noisy defocus must not overwrite the focused stroke"
    );
}

#[test]
fn diffuse_halo_cannot_outvote_a_narrow_focused_contour() {
    let sharp = Rgb32FImage::from_fn(192, 160, |x, y| {
        let line = (x as i32 - 96).abs() <= 1 && (20..140).contains(&y);
        let paper = 0.40 + 0.003 * (x as f32 * 0.17).sin();
        let value = if line { 0.28 } else { paper };
        Rgb([value, value * 0.9, value * 0.8])
    });
    let base = image::imageops::blur(&sharp, 5.0);
    let info = image_info(1, &sharp);
    let sampler = LayerSampler {
        info: &info,
        source: &sharp,
        source_divisor: 1.0,
        inverse: Matrix3::identity(),
        native_mapping: None,
        projection: Projection::Planar,
        offset: (0.0, 0.0),
        left: 0,
        top: 0,
        scale: 1.0,
        residual: Field::new(192, 160, 48.0),
    };
    let chosen = ownership(
        &base,
        &GrayImage::from_pixel(192, 160, Luma([255])),
        &sampler,
        &Field::new(192, 160, 56.0),
        192,
        160,
    );
    let selected = (30..130)
        .filter(|&y| chosen.get_pixel(96, y)[0] > 0)
        .count();
    assert!(
        selected >= 90,
        "focused contour lost to its spread halo: {selected}/100"
    );
}

#[test]
fn enclosed_sharp_tile_replaces_blur_without_losing_coverage() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    // Four times the old 160x120 fixture, with the candidate a 320x320 crop
    // instead of an 80x80 one.  On the small canvas the native control
    // points had to sit at least one 57x57 patch away from the layer edge,
    // which left 2 measurable cells out of 18 — an 11% inlier spatial
    // support that is an artefact of an 80px layer, not of the photographs,
    // and that 需求 2.8 / 12.2 correctly refuse to fuse.  A 320px layer
    // measures 0.4 of its cells, so the test exercises the selection it is
    // about; the exclusion path has its own unit tests in
    // `stack_pipeline::station_degradation`.
    let sharp = Rgb32FImage::from_fn(640, 480, |x, y| {
        let value = 0.36
            + 0.12 * (x as f32 * 0.75).sin()
            + 0.10 * (y as f32 * 0.65).cos()
            + 0.03 * ((x + y) as f32 * 0.19).sin();
        Rgb([value * 1.1, value, value * 0.8])
    });
    let base = image::imageops::blur(&sharp, 2.0);
    let candidate = image::imageops::crop_imm(&sharp, 160, 80, 320, 320).to_image();
    let sources = [base.clone(), candidate];
    let infos = [image_info(0, &sources[0]), image_info(1, &sources[1])];
    let transforms = HashMap::from([
        (0, Matrix3::identity()),
        (
            1,
            Matrix3::new(1.0, 0.0, 160.0, 0.0, 1.0, 80.0, 0.0, 0.0, 1.0),
        ),
    ]);
    let app = tauri::test::mock_app();
    let output = detail_preserving_mosaic(
        &[&infos[0], &infos[1]],
        &transforms,
        Projection::Planar,
        None,
        false,
        MosaicToneFinish::ContentLevelling,
        app.handle().clone(),
        "test-progress",
        &mut |info| Ok(sources[info.id].clone()),
    )
    .unwrap();
    assert_eq!(
        output.dimensions(),
        sharp.dimensions(),
        "coverage must include the last row and column"
    );
    let mut before = 0.0;
    let mut after = 0.0;
    for y in 120..360 {
        for x in 200..440 {
            before += (base.get_pixel(x, y)[1] - sharp.get_pixel(x, y)[1]).powi(2);
            after += (output.get_pixel(x, y)[1] - sharp.get_pixel(x, y)[1]).powi(2);
        }
    }
    assert!(
        after < before * 0.05,
        "the sharp interior must replace the blur: {after} vs {before}"
    );
    assert!(
        output
            .pixels()
            .all(|p| p.0.iter().all(|v| v.is_finite() && *v > 0.0))
    );
}

#[test]
fn blurred_candidate_cannot_replace_an_already_sharp_interior() {
    let base = sharp_fixture();
    let source = image::imageops::blur(&base, 2.0);
    let info = image_info(0, &source);
    let sampler = LayerSampler {
        info: &info,
        source: &source,
        source_divisor: 1.0,
        inverse: Matrix3::identity(),
        native_mapping: None,
        projection: Projection::Planar,
        offset: (0.0, 0.0),
        left: 0,
        top: 0,
        scale: 1.0,
        residual: Field::new(160, 120, 48.0),
    };
    let mask = GrayImage::from_pixel(160, 120, Luma([255]));
    let selected = ownership(
        &base,
        &mask,
        &sampler,
        &Field::new(160, 120, 56.0),
        160,
        120,
    );
    let switched = (10..110)
        .flat_map(|y| (10..150).map(move |x| (x, y)))
        .filter(|&(x, y)| selected.get_pixel(x, y)[0] != 0)
        .count();
    assert_eq!(switched, 0);
}

#[test]
fn sparse_in_focus_stroke_is_not_lost_to_cell_background() {
    let sharp = Rgb32FImage::from_fn(160, 120, |x, y| {
        let line = (x as i32 - 80).unsigned_abs() <= 1 && (12..108).contains(&y);
        let value = if line { 0.04 } else { 0.42 };
        Rgb([value, value, value])
    });
    let base = image::imageops::blur(&sharp, 3.0);
    let info = image_info(1, &sharp);
    let sampler = LayerSampler {
        info: &info,
        source: &sharp,
        source_divisor: 1.0,
        inverse: Matrix3::identity(),
        native_mapping: None,
        projection: Projection::Planar,
        offset: (0.0, 0.0),
        left: 0,
        top: 0,
        scale: 1.0,
        residual: Field::new(160, 120, 48.0),
    };
    let mask = GrayImage::from_pixel(160, 120, Luma([255]));
    let selected = ownership(
        &base,
        &mask,
        &sampler,
        &Field::new(160, 120, 56.0),
        160,
        120,
    );
    let selected_on_stroke = (12..108)
        .filter(|&y| selected.get_pixel(80, y)[0] != 0)
        .count();
    assert!(
        selected_on_stroke > 70,
        "a sparse sharp stroke must own its pixels ({selected_on_stroke}/96)"
    );
}

#[test]
fn streaming_focus_probe_keeps_a_sparse_sharp_stroke() {
    let sharp = Rgb32FImage::from_fn(160, 128, |x, y| {
        let line = (x as i32 - 80).unsigned_abs() <= 1 && (8..120).contains(&y);
        let value = if line { 0.04 } else { 0.42 };
        Rgb([value, value, value])
    });
    let base = image::imageops::blur(&sharp, 3.0);
    let info = image_info(1, &sharp);
    let sampler = LayerSampler {
        info: &info,
        source: &sharp,
        source_divisor: 1.0,
        inverse: Matrix3::identity(),
        native_mapping: None,
        projection: Projection::Planar,
        offset: (0.0, 0.0),
        left: 0,
        top: 0,
        scale: 1.0,
        residual: Field::new(160, 128, 48.0),
    };
    let selected = streaming_ownership_with_long_side(
        &base,
        &GrayImage::from_pixel(160, 128, Luma([255])),
        &sampler,
        &Field::new(160, 128, 56.0),
        160,
        128,
        5,
    );
    assert!(
        selected.get_pixel(2, 1)[0] != 0 || selected.get_pixel(2, 2)[0] != 0,
        "the bounded streaming probe must retain sparse focused writing"
    );
}

#[test]
fn disagreement_is_lower_for_aligned_defocus_than_for_shifted_detail() {
    let sharp = Rgb32FImage::from_fn(160, 120, |x, y| {
        let value = 0.35 + 0.22 * (x as f32 * 0.37).sin() + 0.16 * (y as f32 * 0.29).cos();
        Rgb([value, value * 0.9, value * 0.75])
    });
    let base = image::imageops::blur(&sharp, 2.5);
    let info = image_info(1, &sharp);
    let tone = Field::new(160, 120, 56.0);
    let aligned = LayerSampler {
        info: &info,
        source: &sharp,
        source_divisor: 1.0,
        inverse: Matrix3::identity(),
        native_mapping: None,
        projection: Projection::Planar,
        offset: (0.0, 0.0),
        left: 0,
        top: 0,
        scale: 1.0,
        residual: Field::new(160, 120, 48.0),
    };
    let aligned_error = ownership_disagreement(&base, &aligned, &tone, 32.0, 32.0, 40.0);
    let shifted = LayerSampler {
        info: &info,
        source: &sharp,
        source_divisor: 1.0,
        inverse: Matrix3::new(1.0, 0.0, 4.0, 0.0, 1.0, -3.0, 0.0, 0.0, 1.0),
        native_mapping: None,
        projection: Projection::Planar,
        offset: (0.0, 0.0),
        left: 0,
        top: 0,
        scale: 1.0,
        residual: Field::new(160, 120, 48.0),
    };
    let shifted_error = ownership_disagreement(&base, &shifted, &tone, 32.0, 32.0, 40.0);
    assert!(
        aligned_error < shifted_error,
        "local averaging should preserve geometric mismatch signal: {aligned_error} vs {shifted_error}"
    );
}

#[test]
fn invalid_sample_gap_cannot_cancel_disconnected_colour_mismatches() {
    let base = Rgb32FImage::from_pixel(100, 100, Rgb([0.5; 3]));
    let source = Rgb32FImage::from_fn(100, 100, |x, _| {
        // Invalid samples must retain their positions. If the valid
        // samples are packed into a smaller grid, these separated light
        // and dark strips become neighbours and falsely cancel out.
        let value = if x < 30 {
            0.8
        } else if x > 60 {
            0.2
        } else {
            f32::NAN
        };
        Rgb([value; 3])
    });
    let info = image_info(1, &source);
    let sampler = LayerSampler {
        info: &info,
        source: &source,
        source_divisor: 1.0,
        inverse: Matrix3::identity(),
        native_mapping: None,
        projection: Projection::Planar,
        offset: (0.0, 0.0),
        left: 0,
        top: 0,
        scale: 1.0,
        residual: Field::new(100, 100, 48.0),
    };
    let error =
        ownership_disagreement(&base, &sampler, &Field::new(100, 100, 56.0), 0.0, 0.0, 80.0);
    assert!(
        (error - 0.3).abs() < 1e-6,
        "missing samples must not hide a real colour mismatch: {error}"
    );
}

#[test]
fn fully_invalid_overlap_is_not_treated_as_an_aligned_match() {
    let source = Rgb32FImage::from_pixel(32, 32, Rgb([0.5; 3]));
    let info = image_info(1, &source);
    let sampler = LayerSampler {
        info: &info,
        source: &source,
        source_divisor: 1.0,
        inverse: Matrix3::new(1.0, 0.0, 64.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        native_mapping: None,
        projection: Projection::Planar,
        offset: (0.0, 0.0),
        left: 0,
        top: 0,
        scale: 1.0,
        residual: Field::new(32, 32, 48.0),
    };
    assert_eq!(
        ownership_disagreement(&source, &sampler, &Field::new(32, 32, 56.0), 0.0, 0.0, 24.0,),
        1.0,
        "absence of common samples must not remove the mismatch penalty"
    );
}

#[test]
fn scaled_sparse_detail_is_sampled_at_the_native_cell_centre() {
    let sharp = Rgb32FImage::from_fn(3_200, 1_600, |x, _| {
        let value = if (x as i32 - 84).unsigned_abs() <= 2 {
            0.04
        } else {
            0.42
        };
        Rgb([value; 3])
    });
    let base = image::imageops::blur(&sharp, 4.0);
    let info = image_info(1, &sharp);
    let sampler = LayerSampler {
        info: &info,
        source: &sharp,
        source_divisor: 1.0,
        inverse: Matrix3::identity(),
        native_mapping: None,
        projection: Projection::Planar,
        offset: (0.0, 0.0),
        left: 0,
        top: 0,
        scale: 8.0,
        residual: Field::new(400, 200, 48.0),
    };
    let selected = ownership(
        &base,
        &GrayImage::from_pixel(3_200, 1_600, Luma([255])),
        &sampler,
        &Field::new(400, 200, 56.0),
        400,
        200,
    );
    let selected_on_stroke = (20..180)
        .filter(|&y| selected.get_pixel(10, y)[0] != 0)
        .count();
    assert!(
        selected_on_stroke > 100,
        "a native sharp stroke at a scaled cell centre must own its detail ({selected_on_stroke}/160)"
    );
}

#[test]
fn streaming_scan_lets_the_sharper_focus_bracket_win_inside_its_station() {
    // Two focus brackets taken at one camera position of a moving scan. A
    // scan labels canvas owners by source; the second bracket used to be
    // routed through a panorama seam as if it were a new position, so the
    // blurred first bracket kept every pixel both of them covered.
    let sharp = Rgb32FImage::from_fn(640, 480, |x, y| {
        let value = 0.36
            + 0.12 * (x as f32 * 0.75).sin()
            + 0.10 * (y as f32 * 0.65).cos()
            + 0.03 * ((x + y) as f32 * 0.19).sin();
        Rgb([value * 1.1, value, value * 0.8])
    });
    let blurred = image::imageops::blur(&sharp, 2.0);
    let sources = [blurred.clone(), sharp.clone()];
    let infos = [image_info(0, &sources[0]), image_info(1, &sources[1])];
    let transforms = HashMap::from([(0, Matrix3::identity()), (1, Matrix3::identity())]);
    let stations = HashMap::from([(0, 1u8), (1, 1u8)]);
    let app = tauri::test::mock_app();
    let output = detail_preserving_mosaic_streaming(
        &[&infos[0], &infos[1]],
        &transforms,
        Projection::Planar,
        Some(&stations),
        true,
        MosaicToneFinish::OverlapMeasured,
        app.handle().clone(),
        "test-progress",
        &mut |info| Ok(sources[info.id].clone()),
        0.0,
        0.0,
        640,
        480,
    )
    .unwrap();
    assert_eq!(output.dimensions(), sharp.dimensions());
    let mut blurred_error = 0.0;
    let mut output_error = 0.0;
    for y in 80..400 {
        for x in 80..560 {
            blurred_error += (blurred.get_pixel(x, y)[1] - sharp.get_pixel(x, y)[1]).powi(2);
            output_error += (output.get_pixel(x, y)[1] - sharp.get_pixel(x, y)[1]).powi(2);
        }
    }
    assert!(
        output_error < blurred_error * 0.25,
        "the sharper bracket must own its station's interior: {output_error} vs {blurred_error}"
    );
}
