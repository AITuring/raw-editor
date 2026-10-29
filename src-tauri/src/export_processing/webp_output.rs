//! Bounded-memory WebP output.
//!
//! Encodes through libwebp straight into the output file and rewrites the ICC
//! chunk with a fixed copy buffer, honouring cancellation throughout.

use crate::color_management::srgb_v4_profile;
use image::DynamicImage;
use std::fs;
use std::io::Read;
#[cfg(not(target_os = "android"))]
use std::io::Seek;
#[cfg(not(target_os = "android"))]
use std::io::SeekFrom;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use super::ensure_export_not_cancelled;
#[cfg(not(target_os = "android"))]
use super::{create_temporary_export, preserve_existing_export_permissions};

#[cfg(not(target_os = "android"))]
pub(super) const WEBP_OUTPUT_COPY_BUFFER_BYTES: usize = 64 * 1024;

#[cfg(not(target_os = "android"))]
pub(super) struct WebpPictureGuard(pub(super) libwebp_sys::WebPPicture);

#[cfg(not(target_os = "android"))]
impl Drop for WebpPictureGuard {
    fn drop(&mut self) {
        // SAFETY: WebPPicture::new initialized the value and this guard owns
        // the sole call that releases its libwebp allocations.
        unsafe { libwebp_sys::WebPPictureFree(&mut self.0) };
    }
}

#[cfg(not(target_os = "android"))]
pub(super) struct WebpFileWriterContext {
    pub(super) output: *mut fs::File,
    pub(super) cancellation_token: *const AtomicBool,
    pub(super) cancelled: bool,
    pub(super) first_error: Option<String>,
}

#[cfg(not(target_os = "android"))]
unsafe extern "C" fn write_webp_bytes_to_file(
    data: *const u8,
    data_size: usize,
    picture: *const libwebp_sys::WebPPicture,
) -> std::ffi::c_int {
    let completed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if picture.is_null() {
            return false;
        }

        // SAFETY: encode_webp_to_file sets custom_ptr to this stack context
        // immediately before the synchronous WebPEncode call and clears it
        // before returning.
        let context_ptr = unsafe { (*picture).custom_ptr.cast::<WebpFileWriterContext>() };
        if context_ptr.is_null() {
            return false;
        }
        // SAFETY: the pointer remains valid and exclusively used by libwebp
        // for the duration of the synchronous encoder call.
        let context = unsafe { &mut *context_ptr };
        if data_size == 0 {
            return true;
        }
        if data.is_null() || context.output.is_null() {
            context
                .first_error
                .get_or_insert_with(|| "libwebp supplied an invalid output buffer".to_string());
            return false;
        }

        // SAFETY: libwebp guarantees data points to data_size readable bytes
        // for the duration of this callback.
        let bytes = unsafe { std::slice::from_raw_parts(data, data_size) };
        // SAFETY: output points to the uniquely borrowed File held by the
        // callback context for the duration of WebPEncode.
        let output = unsafe { &mut *context.output };
        if let Err(error) = output.write_all(bytes) {
            context
                .first_error
                .get_or_insert_with(|| format!("Failed to write WebP output: {error}"));
            return false;
        }
        true
    }))
    .unwrap_or(false);

    i32::from(completed)
}

#[cfg(not(target_os = "android"))]
pub(super) unsafe extern "C" fn check_webp_encoding_progress(
    _percent: std::ffi::c_int,
    picture: *const libwebp_sys::WebPPicture,
) -> std::ffi::c_int {
    let should_continue = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if picture.is_null() {
            return false;
        }
        // SAFETY: encode_webp_to_file installs the same live callback context
        // in user_data for the duration of the synchronous WebPEncode call.
        let context_ptr = unsafe { (*picture).user_data.cast::<WebpFileWriterContext>() };
        if context_ptr.is_null() {
            return false;
        }
        // SAFETY: libwebp invokes the progress hook synchronously and the
        // context remains valid until WebPEncode returns.
        let context = unsafe { &mut *context_ptr };
        if context.cancellation_token.is_null() {
            return true;
        }
        // SAFETY: the caller-owned cancellation token outlives WebPEncode and
        // AtomicBool permits concurrent reads from the export worker.
        if unsafe { &*context.cancellation_token }.load(Ordering::SeqCst) {
            context.cancelled = true;
            return false;
        }
        true
    }))
    .unwrap_or(false);

    i32::from(should_continue)
}

#[cfg(not(target_os = "android"))]
fn encode_webp_to_file(
    image: &DynamicImage,
    jpeg_quality: u8,
    output: &mut fs::File,
    cancellation_token: &AtomicBool,
) -> Result<(), String> {
    const WEBP_MAX_DIMENSION: u32 = 16_383;

    let (pixels, width, height, bytes_per_pixel) = match image {
        DynamicImage::ImageRgb8(image) => (
            image.as_raw().as_slice(),
            image.width(),
            image.height(),
            3_usize,
        ),
        DynamicImage::ImageRgba8(image) => (
            image.as_raw().as_slice(),
            image.width(),
            image.height(),
            4_usize,
        ),
        _ => return Err("Failed to create WebP encoder".to_string()),
    };
    if width == 0 || height == 0 || width > WEBP_MAX_DIMENSION || height > WEBP_MAX_DIMENSION {
        return Err(format!(
            "WebP dimensions {width}x{height} exceed the format limit of {WEBP_MAX_DIMENSION} pixels per axis"
        ));
    }
    let expected_len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(bytes_per_pixel))
        .ok_or_else(|| "WebP input dimensions exceed the addressable buffer size".to_string())?;
    if pixels.len() != expected_len {
        return Err(format!(
            "WebP input has {} bytes; expected {expected_len}",
            pixels.len()
        ));
    }
    let stride = (width as usize)
        .checked_mul(bytes_per_pixel)
        .and_then(|value| i32::try_from(value).ok())
        .ok_or_else(|| "WebP row stride exceeds the encoder limit".to_string())?;

    let mut config = libwebp_sys::WebPConfig::new()
        .map_err(|_| "Failed to initialize the WebP encoder configuration".to_string())?;
    config.lossless = 0;
    config.alpha_compression = 1;
    config.quality = f32::from(jpeg_quality);
    // SAFETY: config was initialized by libwebp and remains valid here.
    if unsafe { libwebp_sys::WebPValidateConfig(&config) } == 0 {
        return Err("Invalid WebP encoder configuration".to_string());
    }

    let picture = libwebp_sys::WebPPicture::new()
        .map_err(|_| "Failed to initialize the WebP picture".to_string())?;
    let mut picture = WebpPictureGuard(picture);
    // Lossy WebP ultimately encodes YUV. Import directly into libwebp's YUVA
    // picture instead of retaining a second full ARGB frame until encoding.
    picture.0.use_argb = 0;
    picture.0.width = i32::try_from(width).map_err(|error| error.to_string())?;
    picture.0.height = i32::try_from(height).map_err(|error| error.to_string())?;
    // SAFETY: the validated input slice contains exactly height rows of the
    // declared stride and stays alive until the synchronous import completes.
    let imported = unsafe {
        if bytes_per_pixel == 4 {
            libwebp_sys::WebPPictureImportRGBA(&mut picture.0, pixels.as_ptr(), stride)
        } else {
            libwebp_sys::WebPPictureImportRGB(&mut picture.0, pixels.as_ptr(), stride)
        }
    };
    if imported == 0 {
        return Err(format!(
            "Failed to import WebP pixels: {:?}",
            picture.0.error_code
        ));
    }

    let mut writer_context = WebpFileWriterContext {
        output,
        cancellation_token,
        cancelled: false,
        first_error: None,
    };
    picture.0.writer = Some(write_webp_bytes_to_file);
    picture.0.custom_ptr = std::ptr::from_mut(&mut writer_context).cast();
    picture.0.progress_hook = Some(check_webp_encoding_progress);
    picture.0.user_data = std::ptr::from_mut(&mut writer_context).cast();
    // SAFETY: config and picture are initialized; all pixel data, callback
    // state and output storage outlive this synchronous call.
    let status = unsafe { libwebp_sys::WebPEncode(&config, &mut picture.0) };
    picture.0.writer = None;
    picture.0.custom_ptr = std::ptr::null_mut();
    picture.0.progress_hook = None;
    picture.0.user_data = std::ptr::null_mut();

    if status == 0 {
        if writer_context.cancelled {
            return Err("Export cancelled".to_string());
        }
        return Err(writer_context.first_error.unwrap_or_else(|| {
            format!("Failed to encode WebP image: {:?}", picture.0.error_code)
        }));
    }
    if let Some(error) = writer_context.first_error {
        return Err(error);
    }
    Ok(())
}

#[cfg(not(target_os = "android"))]
fn inspect_webp_file<R: Read + Seek>(
    source: &mut R,
    cancellation_token: &AtomicBool,
) -> Result<(u64, bool), String> {
    ensure_export_not_cancelled(cancellation_token)?;
    let source_len = source
        .seek(SeekFrom::End(0))
        .map_err(|error| format!("Failed to inspect WebP output length: {error}"))?;
    source
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("Failed to rewind WebP output: {error}"))?;

    let mut riff_header = [0_u8; 12];
    source
        .read_exact(&mut riff_header)
        .map_err(|_| "WebP encoder returned an invalid RIFF container".to_string())?;
    if &riff_header[..4] != b"RIFF" || &riff_header[8..] != b"WEBP" {
        return Err("WebP encoder returned an invalid RIFF container".to_string());
    }
    let declared_len = u64::from(u32::from_le_bytes(
        riff_header[4..8]
            .try_into()
            .map_err(|_| "Invalid WebP RIFF length".to_string())?,
    )) + 8;
    if declared_len != source_len {
        return Err(format!(
            "WebP RIFF length {declared_len} does not match file length {source_len}"
        ));
    }

    let mut offset = 12_u64;
    let mut found_vp8x = false;
    while offset + 8 <= source_len {
        ensure_export_not_cancelled(cancellation_token)?;
        let mut chunk_header = [0_u8; 8];
        source
            .read_exact(&mut chunk_header)
            .map_err(|error| format!("Failed to read WebP chunk header: {error}"))?;
        let chunk_len = u64::from(u32::from_le_bytes(
            chunk_header[4..8]
                .try_into()
                .map_err(|_| "Invalid WebP chunk length".to_string())?,
        ));
        let padded_len = chunk_len + chunk_len % 2;
        let chunk_end = offset
            .checked_add(8)
            .and_then(|value| value.checked_add(padded_len))
            .ok_or_else(|| "WebP chunk size overflow".to_string())?;
        if chunk_end > source_len {
            return Err("WebP chunk extends past the RIFF container".to_string());
        }
        if &chunk_header[..4] == b"VP8X" {
            if chunk_len != 10 {
                return Err("WebP VP8X chunk has an invalid length".to_string());
            }
            found_vp8x = true;
        }
        source
            .seek(SeekFrom::Current(padded_len as i64))
            .map_err(|error| format!("Failed to seek over WebP chunk: {error}"))?;
        offset = chunk_end;
    }
    if offset != source_len {
        return Err("WebP container has trailing partial chunk data".to_string());
    }
    Ok((source_len, found_vp8x))
}

#[cfg(not(target_os = "android"))]
fn write_webp_chunk_to_writer<W: Write>(
    output: &mut W,
    fourcc: &[u8; 4],
    payload: &[u8],
) -> Result<u64, String> {
    let payload_len =
        u32::try_from(payload.len()).map_err(|_| "WebP chunk is too large for RIFF".to_string())?;
    output
        .write_all(fourcc)
        .and_then(|_| output.write_all(&payload_len.to_le_bytes()))
        .and_then(|_| output.write_all(payload))
        .map_err(|error| format!("Failed to write WebP chunk: {error}"))?;
    if !payload.len().is_multiple_of(2) {
        output
            .write_all(&[0])
            .map_err(|error| format!("Failed to pad WebP chunk: {error}"))?;
    }
    Ok(8 + payload.len() as u64 + (payload.len() % 2) as u64)
}

#[cfg(not(target_os = "android"))]
fn copy_webp_bytes_bounded<R: Read, W: Write>(
    source: &mut R,
    output: &mut W,
    mut remaining: u64,
    buffer: &mut [u8; WEBP_OUTPUT_COPY_BUFFER_BYTES],
    cancellation_token: &AtomicBool,
) -> Result<(), String> {
    while remaining > 0 {
        ensure_export_not_cancelled(cancellation_token)?;
        let chunk_len = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|error| error.to_string())?;
        source
            .read_exact(&mut buffer[..chunk_len])
            .map_err(|error| format!("Failed to read encoded WebP data: {error}"))?;
        output
            .write_all(&buffer[..chunk_len])
            .map_err(|error| format!("Failed to rewrite encoded WebP data: {error}"))?;
        remaining -= chunk_len as u64;
    }
    Ok(())
}

#[cfg(not(target_os = "android"))]
pub(super) fn rewrite_webp_icc_bounded<R: Read + Seek, W: Write + Seek>(
    source: &mut R,
    output: &mut W,
    icc_profile: &[u8],
    width: u32,
    height: u32,
    has_alpha: bool,
    cancellation_token: &AtomicBool,
) -> Result<u64, String> {
    if width == 0 || height == 0 || width > 0x01_00_00_00 || height > 0x01_00_00_00 {
        return Err("WebP dimensions cannot be represented by a VP8X header".to_string());
    }
    let (source_len, found_vp8x) = inspect_webp_file(source, cancellation_token)?;
    source
        .seek(SeekFrom::Start(12))
        .map_err(|error| format!("Failed to rewind WebP chunks: {error}"))?;
    output
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("Failed to initialize rewritten WebP output: {error}"))?;
    output
        .write_all(b"RIFF\0\0\0\0WEBP")
        .map_err(|error| format!("Failed to write WebP RIFF header: {error}"))?;
    let mut output_len = 12_u64;
    let mut wrote_icc = false;
    if !found_vp8x {
        let mut vp8x = [0_u8; 10];
        vp8x[0] = 0x20 | if has_alpha { 0x10 } else { 0 };
        vp8x[4..7].copy_from_slice(&(width - 1).to_le_bytes()[..3]);
        vp8x[7..10].copy_from_slice(&(height - 1).to_le_bytes()[..3]);
        output_len += write_webp_chunk_to_writer(output, b"VP8X", &vp8x)?;
        output_len += write_webp_chunk_to_writer(output, b"ICCP", icc_profile)?;
        wrote_icc = true;
    }

    let mut offset = 12_u64;
    let mut copy_buffer = [0_u8; WEBP_OUTPUT_COPY_BUFFER_BYTES];
    while offset + 8 <= source_len {
        ensure_export_not_cancelled(cancellation_token)?;
        let mut chunk_header = [0_u8; 8];
        source
            .read_exact(&mut chunk_header)
            .map_err(|error| format!("Failed to read WebP chunk header: {error}"))?;
        let fourcc: [u8; 4] = chunk_header[..4]
            .try_into()
            .map_err(|_| "Invalid WebP chunk identifier".to_string())?;
        let chunk_len = u64::from(u32::from_le_bytes(
            chunk_header[4..8]
                .try_into()
                .map_err(|_| "Invalid WebP chunk length".to_string())?,
        ));
        let padded_len = chunk_len + chunk_len % 2;

        if &fourcc == b"ICCP" {
            source
                .seek(SeekFrom::Current(padded_len as i64))
                .map_err(|error| format!("Failed to skip existing WebP ICC chunk: {error}"))?;
        } else if &fourcc == b"VP8X" {
            let mut payload = [0_u8; 10];
            source
                .read_exact(&mut payload)
                .map_err(|error| format!("Failed to read WebP VP8X chunk: {error}"))?;
            payload[0] |= 0x20;
            output_len += write_webp_chunk_to_writer(output, b"VP8X", &payload)?;
            if !wrote_icc {
                output_len += write_webp_chunk_to_writer(output, b"ICCP", icc_profile)?;
                wrote_icc = true;
            }
        } else {
            output
                .write_all(&chunk_header)
                .map_err(|error| format!("Failed to write WebP chunk header: {error}"))?;
            copy_webp_bytes_bounded(
                source,
                output,
                padded_len,
                &mut copy_buffer,
                cancellation_token,
            )?;
            output_len = output_len
                .checked_add(8 + padded_len)
                .ok_or_else(|| "WebP output size overflow".to_string())?;
        }
        offset += 8 + padded_len;
    }
    if !wrote_icc {
        return Err("Failed to insert the WebP ICC profile".to_string());
    }

    let riff_size = u32::try_from(
        output_len
            .checked_sub(8)
            .ok_or_else(|| "WebP output size underflow".to_string())?,
    )
    .map_err(|_| "WebP output is too large for RIFF".to_string())?;
    output
        .seek(SeekFrom::Start(4))
        .and_then(|_| output.write_all(&riff_size.to_le_bytes()))
        .and_then(|_| output.seek(SeekFrom::Start(output_len)).map(|_| ()))
        .map_err(|error| format!("Failed to finalize WebP RIFF output: {error}"))?;
    ensure_export_not_cancelled(cancellation_token)?;
    Ok(output_len)
}

#[cfg(not(target_os = "android"))]
pub(super) fn save_webp_with_bounded_output(
    image: &DynamicImage,
    output_path: &Path,
    jpeg_quality: u8,
    cancellation_token: &AtomicBool,
) -> Result<(), String> {
    ensure_export_not_cancelled(cancellation_token)?;
    let mut encoded_temporary = create_temporary_export(output_path)?;
    encode_webp_to_file(
        image,
        jpeg_quality,
        encoded_temporary.as_file_mut(),
        cancellation_token,
    )?;
    encoded_temporary
        .as_file_mut()
        .flush()
        .map_err(|error| format!("Failed to flush encoded WebP output: {error}"))?;
    ensure_export_not_cancelled(cancellation_token)?;

    let mut final_temporary = create_temporary_export(output_path)?;
    rewrite_webp_icc_bounded(
        encoded_temporary.as_file_mut(),
        final_temporary.as_file_mut(),
        srgb_v4_profile(),
        image.width(),
        image.height(),
        image.color().has_alpha(),
        cancellation_token,
    )?;
    final_temporary
        .as_file_mut()
        .flush()
        .map_err(|error| format!("Failed to flush rewritten WebP output: {error}"))?;
    ensure_export_not_cancelled(cancellation_token)?;
    drop(encoded_temporary);
    preserve_existing_export_permissions(&final_temporary, output_path)?;
    final_temporary.persist(output_path).map_err(|error| {
        format!(
            "Failed to atomically publish WebP export '{}': {}",
            output_path.display(),
            error.error
        )
    })?;
    Ok(())
}
