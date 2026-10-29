//! Whole entry-point runs on small fixtures: an undecodable source, the synthetic 200-image
//! scalable path and the real three-image panorama.

use super::*;

fn synthetic_texture_pixel(x: u32, y: u32) -> Rgb<u8> {
    let mut hash = x.wrapping_mul(0x9E37_79B9) ^ y.wrapping_mul(0x85EB_CA6B);
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x7FEB_352D);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x846C_A68B);
    hash ^= hash >> 16;
    let value = hash as u8;
    Rgb([
        value,
        value.rotate_left(3) ^ ((x.wrapping_add(y) * 13) as u8),
        value.rotate_right(2) ^ ((x.wrapping_mul(3).wrapping_add(y * 5)) as u8),
    ])
}

/// 需求 1.9 excludes an undecodable Source_RAW and continues; 需求 1.8 then
/// rejects the run when fewer than two decodable sources are left.  The
/// panorama path shares this entry point and keeps its original behaviour:
/// it has no Capture_Station layer to exclude a source from, so it still
/// aborts on the first preparation error.
#[test]
// Like every other test that drives the whole entry point, this one needs the
// process-global sidecar storage, whose `OnceCell` cannot be shared with
// `sidecar_storage`'s own tests inside one test binary.
#[ignore = "initializes the process-global sidecar storage; run explicitly"]
fn undecodable_source_is_excluded_before_the_input_count_is_rejected() {
    let fixture_dir = tempfile::tempdir().expect("temporary stack fixture should be writable");
    let readable = fixture_dir.path().join("tile-000.png");
    RgbImage::from_fn(96, 96, synthetic_texture_pixel)
        .save_with_format(&readable, ImageFormat::Png)
        .expect("synthetic stack tile should be writable");
    let broken = fixture_dir.path().join("tile-001.png");
    std::fs::write(&broken, b"not an image").expect("broken fixture should be writable");
    let paths = vec![
        readable.to_string_lossy().into_owned(),
        broken.to_string_lossy().into_owned(),
    ];

    // Process global, set once by whichever test gets there first; any
    // initialized root is usable here because this run never reaches a
    // sidecar write.
    let _ = crate::sidecar_storage::initialize(
        &std::env::temp_dir().join("raw-editor-stack-decode-test-sidecars"),
    );
    let app = tauri::test::mock_app();
    let focus_error = stitch_images_with_options(
        paths.clone(),
        app.handle().clone(),
        AlignmentMode::Position,
        BlendMode::FocusStack,
        "test-image-stack-progress",
    )
    .err()
    .expect("one decodable source is below the 需求 1.8 minimum of two");
    assert!(
        focus_error.contains("Only 1 of 2"),
        "需求 1.9 must exclude the undecodable source and 需求 1.8 must then reject on the \
         surviving count, got: {focus_error}"
    );

    let panorama_error = stitch_images_with_options(
        paths,
        app.handle().clone(),
        AlignmentMode::Position,
        BlendMode::Panorama,
        "test-image-stack-progress",
    )
    .err()
    .expect("the panorama path still aborts on the first preparation error");
    assert!(
        !panorama_error.contains("Only 1 of 2"),
        "the panorama path must keep its original abort, got: {panorama_error}"
    );
}

#[test]
#[ignore = "manual deterministic 200-image scalable stitching smoke test"]
fn synthetic_two_hundred_image_scalable_path() {
    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 240;
    const HORIZONTAL_STEP: u32 = 2;

    let fixture_dir = tempfile::tempdir().expect("temporary stack fixture should be writable");
    let mut paths = Vec::with_capacity(MAX_STITCH_SOURCE_IMAGES);
    for index in 0..MAX_STITCH_SOURCE_IMAGES {
        let horizontal_offset = index as u32 * HORIZONTAL_STEP;
        let image = RgbImage::from_fn(WIDTH, HEIGHT, |x, y| {
            synthetic_texture_pixel(x + horizontal_offset, y)
        });
        let path = fixture_dir.path().join(format!("tile-{index:03}.png"));
        image
            .save_with_format(&path, ImageFormat::Png)
            .expect("synthetic stack tile should be writable");
        paths.push(path.to_string_lossy().into_owned());
    }

    crate::sidecar_storage::initialize(&fixture_dir.path().join("sidecars"))
        .expect("test sidecar storage should initialize once");
    let app = tauri::test::mock_app();
    let started = Instant::now();
    let result = stitch_images_with_options(
        paths,
        app.handle().clone(),
        AlignmentMode::Position,
        BlendMode::Panorama,
        "test-image-stack-progress",
    )
    .expect("the 200 overlapping tiles should stitch through the scalable path");
    let image = result.image;

    let expected_width = WIDTH + (MAX_STITCH_SOURCE_IMAGES as u32 - 1) * HORIZONTAL_STEP;
    assert!(image.width().abs_diff(expected_width) <= 2);
    assert!(image.height().abs_diff(HEIGHT) <= 2);
    println!(
        "synthetic 200-image stack: {}x{} in {:.2?}",
        image.width(),
        image.height(),
        started.elapsed()
    );
}

#[test]
#[ignore = "requires the real three-image fixture and writes a large temporary output"]
fn real_three_image_panorama_fixture() {
    let fixture_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../src/assets/test")
        .canonicalize()
        .expect("test fixture directory must exist");
    let paths = ["DSC01721_1.tif", "DSC01700.tif", "DSC01728.tif"]
        .iter()
        .map(|name| fixture_root.join(name).to_string_lossy().into_owned())
        .collect();
    let alignment_name = std::env::var("RAW_EDITOR_STACK_ACCEPTANCE_ALIGNMENT")
        .unwrap_or_else(|_| "auto".to_string())
        .to_ascii_lowercase();
    let alignment_mode = AlignmentMode::from_wire(&alignment_name);
    let app = tauri::test::mock_app();
    crate::sidecar_storage::initialize(
        PathBuf::from("/private/tmp/raw-editor-stack-sidecars").as_path(),
    )
    .expect("test sidecar storage should initialize once");
    let result = stitch_images_with_options(
        paths,
        app.handle().clone(),
        alignment_mode,
        BlendMode::Panorama,
        "test-image-stack-progress",
    )
    .expect("the three real overlapping images should stitch");
    let image = result.image;

    assert!(image.width() > 0 && image.height() > 0);
    let output_path = std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(format!(
                "/private/tmp/raw-editor-three-image-panorama-{alignment_name}.tiff"
            ))
        });
    image
        .save_with_format(&output_path, ImageFormat::Tiff)
        .expect("full-resolution TIFF result should be writable");

    let preview = crate::image_processing::downscale_f32_image(&image, 1800, 1800);
    let preview_path = output_path.with_extension("jpg");
    preview
        .save_with_format(&preview_path, ImageFormat::Jpeg)
        .expect("panorama preview should be writable");
    println!(
        "three-image panorama result: {}x{}\nfull: {}\npreview: {}",
        image.width(),
        image.height(),
        output_path.display(),
        preview_path.display()
    );
}
