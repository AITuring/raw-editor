use crate::gpu_processing::Roi;

pub const PREVIEW_PATCH_MAGIC: &[u8; 8] = b"RAWROI01";
pub const PREVIEW_PATCH_HEADER_LEN: usize = PREVIEW_PATCH_MAGIC.len() + 6 * size_of::<u32>();

pub fn normalized_roi_to_pixels(
    normalized: Option<(f32, f32, f32, f32)>,
    full_width: u32,
    full_height: u32,
) -> Option<Roi> {
    let (x, y, width, height) = normalized?;
    if full_width == 0
        || full_height == 0
        || !x.is_finite()
        || !y.is_finite()
        || !width.is_finite()
        || !height.is_finite()
        || width <= 0.0
        || height <= 0.0
    {
        return None;
    }

    let left = x.clamp(0.0, 1.0);
    let top = y.clamp(0.0, 1.0);
    let right = (x + width).clamp(0.0, 1.0);
    let bottom = (y + height).clamp(0.0, 1.0);
    if right <= left || bottom <= top {
        return None;
    }

    let pixel_left = ((left * full_width as f32).floor() as u32).min(full_width - 1);
    let pixel_top = ((top * full_height as f32).floor() as u32).min(full_height - 1);
    let pixel_right =
        ((right * full_width as f32).ceil() as u32).clamp(pixel_left.saturating_add(1), full_width);
    let pixel_bottom = ((bottom * full_height as f32).ceil() as u32)
        .clamp(pixel_top.saturating_add(1), full_height);

    Some(Roi {
        x: pixel_left,
        y: pixel_top,
        width: pixel_right - pixel_left,
        height: pixel_bottom - pixel_top,
    })
}

pub fn encode_preview_patch(
    jpeg_bytes: &[u8],
    patch: Roi,
    full_width: u32,
    full_height: u32,
) -> Vec<u8> {
    let mut response = Vec::with_capacity(PREVIEW_PATCH_HEADER_LEN + jpeg_bytes.len());
    response.extend_from_slice(PREVIEW_PATCH_MAGIC);
    response.extend_from_slice(&patch.x.to_le_bytes());
    response.extend_from_slice(&patch.y.to_le_bytes());
    response.extend_from_slice(&patch.width.to_le_bytes());
    response.extend_from_slice(&patch.height.to_le_bytes());
    response.extend_from_slice(&full_width.to_le_bytes());
    response.extend_from_slice(&full_height.to_le_bytes());
    response.extend_from_slice(jpeg_bytes);
    response
}

#[cfg(test)]
mod tests {
    use super::{
        PREVIEW_PATCH_HEADER_LEN, PREVIEW_PATCH_MAGIC, encode_preview_patch,
        normalized_roi_to_pixels,
    };
    use crate::gpu_processing::Roi;

    #[test]
    fn normalized_roi_uses_covering_pixel_bounds_and_clamps_to_image() {
        let roi = normalized_roi_to_pixels(Some((0.101, -0.2, 0.25, 0.7)), 100, 80).unwrap();

        assert_eq!(roi.x, 10);
        assert_eq!(roi.y, 0);
        assert_eq!(roi.width, 26);
        assert_eq!(roi.height, 40);
    }

    #[test]
    fn normalized_roi_rejects_invalid_or_outside_regions() {
        assert!(normalized_roi_to_pixels(Some((0.0, 0.0, f32::NAN, 1.0)), 100, 80).is_none());
        assert!(normalized_roi_to_pixels(Some((1.2, 0.0, 0.2, 1.0)), 100, 80).is_none());
        assert!(normalized_roi_to_pixels(Some((0.0, 0.0, 1.0, 1.0)), 0, 80).is_none());
    }

    #[test]
    fn preview_patch_response_has_versioned_geometry_header() {
        let jpeg = [0xff, 0xd8, 0xff, 0xd9];
        let response = encode_preview_patch(
            &jpeg,
            Roi {
                x: 11,
                y: 12,
                width: 640,
                height: 480,
            },
            4000,
            3000,
        );

        assert_eq!(&response[..PREVIEW_PATCH_MAGIC.len()], PREVIEW_PATCH_MAGIC);
        assert_eq!(response.len(), PREVIEW_PATCH_HEADER_LEN + jpeg.len());

        let read_u32 =
            |offset: usize| u32::from_le_bytes(response[offset..offset + 4].try_into().unwrap());
        assert_eq!(read_u32(8), 11);
        assert_eq!(read_u32(12), 12);
        assert_eq!(read_u32(16), 640);
        assert_eq!(read_u32(20), 480);
        assert_eq!(read_u32(24), 4000);
        assert_eq!(read_u32(28), 3000);
        assert_eq!(&response[PREVIEW_PATCH_HEADER_LEN..], jpeg);
    }
}
