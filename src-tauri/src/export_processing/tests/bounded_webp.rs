//! Bounded-memory WebP output.
//!
//! Byte parity with the in-memory encoder, ICC chunk rewriting, permission-keeping
//! publication and cancellation.

use super::*;

#[test]
fn bounded_webp_output_matches_the_existing_encoder_bytes() {
    let rgb = fixture_image(37, 23);
    let rgba = DynamicImage::ImageRgba8(RgbaImage::from_fn(37, 23, |x, y| {
        Rgba([
            ((x * 17 + y * 3) % 256) as u8,
            ((x * 5 + y * 13) % 256) as u8,
            ((x * 11 + y * 7) % 256) as u8,
            ((x * 19 + y * 23) % 256) as u8,
        ])
    }));
    let directory = tempfile::tempdir().expect("temporary WebP directory");
    let cancellation_token = AtomicBool::new(false);

    for (index, image) in [rgb, rgba].iter().enumerate() {
        for quality in [50, 75, 90, 100] {
            let expected = encode_image_to_bytes(image, "webp", quality).unwrap_or_else(|error| {
                panic!("encode reference WebP {index} Q{quality}: {error}")
            });
            let output_path = directory
                .path()
                .join(format!("bounded-{index}-q{quality}.webp"));
            save_webp_with_bounded_output(image, &output_path, quality, &cancellation_token)
                .unwrap_or_else(|error| panic!("write bounded WebP {index} Q{quality}: {error}"));
            let actual = fs::read(&output_path).expect("read bounded WebP");

            assert_eq!(actual, expected);
            assert_decoder_contract(
                WebPDecoder::new(Cursor::new(actual)).expect("decode bounded WebP"),
                image.dimensions(),
                true,
            );
        }
    }
}

#[test]
fn bounded_webp_icc_rewrite_replaces_existing_profile_and_rejects_bad_input() {
    let image = fixture_image(37, 23);
    let encoded = encode_image_to_bytes(&image, "webp", 90).expect("encode ICC WebP");
    let cancellation_token = AtomicBool::new(false);
    let mut source = Cursor::new(encoded.clone());
    let mut output = Cursor::new(Vec::new());
    let output_len = rewrite_webp_icc_bounded(
        &mut source,
        &mut output,
        srgb_v4_profile(),
        image.width(),
        image.height(),
        false,
        &cancellation_token,
    )
    .expect("rewrite existing WebP ICC");
    assert_eq!(output_len as usize, encoded.len());
    assert_eq!(output.into_inner(), encoded);
    assert_eq!(WEBP_OUTPUT_COPY_BUFFER_BYTES, 65_536);

    let mut truncated = source.into_inner();
    truncated.pop();
    let error = rewrite_webp_icc_bounded(
        &mut Cursor::new(truncated),
        &mut Cursor::new(Vec::new()),
        srgb_v4_profile(),
        image.width(),
        image.height(),
        false,
        &cancellation_token,
    )
    .expect_err("truncated WebP must fail");
    assert!(error.contains("RIFF length"), "unexpected error: {error}");

    let cancelled = AtomicBool::new(true);
    let error = rewrite_webp_icc_bounded(
        &mut Cursor::new(encoded),
        &mut Cursor::new(Vec::new()),
        srgb_v4_profile(),
        image.width(),
        image.height(),
        false,
        &cancelled,
    )
    .expect_err("cancelled WebP rewrite must fail");
    assert_eq!(error, "Export cancelled");
}

#[cfg(unix)]
#[test]
fn bounded_webp_publish_preserves_permissions_and_cancel_keeps_target() {
    use std::os::unix::fs::PermissionsExt;

    let directory = tempfile::tempdir().expect("temporary WebP export directory");
    let output_path = directory.path().join("existing.webp");
    fs::write(&output_path, b"old").expect("create existing WebP export");
    fs::set_permissions(&output_path, fs::Permissions::from_mode(0o640))
        .expect("set existing permissions");

    let image = fixture_image(37, 23);
    save_webp_with_bounded_output(&image, &output_path, 90, &AtomicBool::new(false))
        .expect("publish bounded WebP");
    let published = fs::read(&output_path).expect("read published WebP");
    let mode = fs::metadata(&output_path)
        .expect("published metadata")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o640);

    let error = save_webp_with_bounded_output(&image, &output_path, 90, &AtomicBool::new(true))
        .expect_err("cancelled WebP publish must fail");
    assert_eq!(error, "Export cancelled");
    assert_eq!(fs::read(&output_path).expect("reread WebP"), published);

    let mut callback_output = tempfile::tempfile().expect("callback output");
    let callback_token = AtomicBool::new(true);
    let mut callback_context = WebpFileWriterContext {
        output: &mut callback_output,
        cancellation_token: &callback_token,
        cancelled: false,
        first_error: None,
    };
    let picture = libwebp_sys::WebPPicture::new().expect("initialize callback picture");
    let mut picture = WebpPictureGuard(picture);
    picture.0.user_data = std::ptr::from_mut(&mut callback_context).cast();
    // SAFETY: the initialized picture points to the live callback context.
    let should_continue = unsafe { check_webp_encoding_progress(1, &picture.0) };
    assert_eq!(should_continue, 0);
    assert!(callback_context.cancelled);
}
