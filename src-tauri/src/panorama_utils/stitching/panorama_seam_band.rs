//! Seam-band blending for the progressive panorama stitcher.
//!
//! The low-frequency pyramid follows the seam band while fine detail stays
//! source-owned; PSD scans blend detail across the measured overlap.

use super::super::coordinates::Projection;
use super::super::coordinates::map_target_to_source;
use crate::panorama_utils::shared::ImageInfo;
use image::GrayImage;
use image::Rgb;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use nalgebra::Point3;
use rayon::prelude::*;

use super::{
    ExposureCompensation, PANORAMA_BLEND_BANDS, PANORAMA_DETAIL_SEAM_FEATHER_RADIUS,
    SeamOrientation, apply_exposure_compensation, get_high_quality_interpolated_pixel,
    multiband_blend, source_sample_has_support,
};

pub(super) fn panorama_detail_alpha(candidate_signed_distance: f32) -> f32 {
    if candidate_signed_distance.is_infinite() {
        return if candidate_signed_distance.is_sign_positive() {
            1.0
        } else {
            0.0
        };
    }
    (0.5 + candidate_signed_distance / (PANORAMA_DETAIL_SEAM_FEATHER_RADIUS * 2.0)).clamp(0.0, 1.0)
}

pub(super) struct SeamBandBlend<'a> {
    pub(super) panorama: &'a mut Rgb32FImage,
    pub(super) panorama_mask: &'a mut GrayImage,
    pub(super) img_to_add_info: &'a ImageInfo,
    pub(super) img_to_add: &'a Rgb32FImage,
    pub(super) h_add: &'a Matrix3<f64>,
    pub(super) projection: Projection,
    pub(super) offset_x: f64,
    pub(super) offset_y: f64,
    pub(super) orientation: SeamOrientation,
    pub(super) seam_coords: &'a [i32],
    pub(super) new_image_is_dominant_side: bool,
    pub(super) min_x: u32,
    pub(super) max_x: u32,
    pub(super) min_y: u32,
    pub(super) max_y: u32,
    pub(super) exposure: &'a ExposureCompensation,
}

pub(super) fn blend_panorama_seam_band(ctx: SeamBandBlend<'_>) {
    let SeamBandBlend {
        panorama,
        panorama_mask,
        img_to_add_info,
        img_to_add,
        h_add,
        projection,
        offset_x,
        offset_y,
        orientation,
        seam_coords,
        new_image_is_dominant_side,
        min_x,
        max_x,
        min_y,
        max_y,
        exposure,
    } = ctx;
    let (out_width, out_height) = panorama.dimensions();
    if out_width == 0 || out_height == 0 || seam_coords.is_empty() {
        return;
    }

    let overlap_left = min_x.min(out_width - 1);
    let overlap_right = max_x.min(out_width - 1);
    let overlap_top = min_y.min(out_height - 1);
    let overlap_bottom = max_y.min(out_height - 1);
    if overlap_left > overlap_right || overlap_top > overlap_bottom {
        return;
    }

    let seam_value = |index: usize| -> Option<u32> {
        seam_coords
            .get(index)
            .copied()
            .map(|value| value.max(0) as u32)
    };

    match orientation {
        SeamOrientation::Horizontal if seam_coords.len() <= overlap_right as usize => return,
        SeamOrientation::Vertical if seam_coords.len() <= overlap_bottom as usize => return,
        _ => {}
    }
    // Keep the complete overlap so the low-frequency mask can transition across the
    // whole shared field rather than inheriting a tonal step around the detail seam.
    let (patch_left, patch_top, patch_right, patch_bottom) =
        (overlap_left, overlap_top, overlap_right, overlap_bottom);

    if patch_left > patch_right || patch_top > patch_bottom {
        return;
    }
    let patch_width = patch_right - patch_left + 1;
    let patch_height = patch_bottom - patch_top + 1;
    if patch_width < 8 || patch_height < 8 {
        return;
    }

    let psd_scan = img_to_add_info
        .filename
        .to_ascii_lowercase()
        .ends_with(".psd");

    let Some(h_add_inv) = h_add.try_inverse() else {
        return;
    };
    let patch_pixel_count = patch_width as usize * patch_height as usize;
    let mut base_pixels = vec![0.0f32; patch_pixel_count * 3];
    let mut candidate_pixels = vec![0.0f32; patch_pixel_count * 3];
    let mut blend_mask = vec![0u8; patch_pixel_count];
    let mut low_frequency_mask = vec![0u8; patch_pixel_count];
    let image_width = img_to_add.width() as f64;
    let image_height = img_to_add.height() as f64;
    let patch_rgb_stride = patch_width as usize * 3;
    let patch_mask_stride = patch_width as usize;
    let panorama_ref: &Rgb32FImage = panorama;
    let panorama_mask_ref: &GrayImage = panorama_mask;

    base_pixels
        .par_chunks_mut(patch_rgb_stride)
        .zip(candidate_pixels.par_chunks_mut(patch_rgb_stride))
        .zip(blend_mask.par_chunks_mut(patch_mask_stride))
        .zip(low_frequency_mask.par_chunks_mut(patch_mask_stride))
        .enumerate()
        .for_each(
            |(local_y, (((base_row, candidate_row), blend_row), low_frequency_row))| {
                let local_y = local_y as u32;
                let global_y = patch_top + local_y;
                for local_x in 0..patch_width {
                    let global_x = patch_left + local_x;
                    let target =
                        Point3::new(global_x as f64 - offset_x, global_y as f64 - offset_y, 1.0);
                    let candidate_source =
                        map_target_to_source(&h_add_inv, target, img_to_add_info, projection);
                    let candidate_valid = candidate_source.as_ref().is_some_and(|source| {
                        source.x >= 0.0
                            && source.x < image_width
                            && source.y >= 0.0
                            && source.y < image_height
                            // A virtual tile is rendered into a rectangular
                            // buffer, but its projective corners are transparent
                            // zero support. Treat those samples as outside the
                            // candidate footprint so they cannot enter either
                            // the low-frequency pyramid or the detail seam.
                            && source_sample_has_support(img_to_add, source.x, source.y)
                    });
                    let candidate_pixel = if let Some(source) = candidate_source {
                        if candidate_valid {
                            apply_exposure_compensation(
                                get_high_quality_interpolated_pixel(img_to_add, source.x, source.y),
                                exposure,
                                global_x,
                                global_y,
                            )
                        } else {
                            Rgb([0.0, 0.0, 0.0])
                        }
                    } else {
                        Rgb([0.0, 0.0, 0.0])
                    };
                    let panorama_valid = panorama_mask_ref.get_pixel(global_x, global_y)[0] > 0;
                    let current_pixel = *panorama_ref.get_pixel(global_x, global_y);
                    let candidate_pixel = if !candidate_valid && panorama_valid {
                        current_pixel
                    } else {
                        candidate_pixel
                    };
                    let base_pixel = if panorama_valid || !candidate_valid {
                        current_pixel
                    } else {
                        candidate_pixel
                    };

                    let candidate_signed_distance = if !candidate_valid {
                        f32::NEG_INFINITY
                    } else if !panorama_valid {
                        f32::INFINITY
                    } else {
                        match orientation {
                            SeamOrientation::Horizontal => {
                                let seam_y = seam_value(global_x as usize).unwrap_or(global_y);
                                let distance = global_y as f32 - seam_y as f32;
                                if new_image_is_dominant_side {
                                    distance
                                } else {
                                    -distance
                                }
                            }
                            SeamOrientation::Vertical => {
                                let seam_x = seam_value(global_y as usize).unwrap_or(global_x);
                                let distance = global_x as f32 - seam_x as f32;
                                if new_image_is_dominant_side {
                                    distance
                                } else {
                                    -distance
                                }
                            }
                        }
                    };
                    let detail_alpha = if psd_scan && candidate_valid && panorama_valid {
                        // A source scan is a planar tile mosaic. Blend its
                        // detail over the complete overlap instead of only a
                        // narrow seam band; a one-pixel pose residual then
                        // cannot leave a source-sized texture rectangle.
                        let span = match orientation {
                            SeamOrientation::Horizontal => {
                                (overlap_bottom - overlap_top).max(1) as f32
                            }
                            SeamOrientation::Vertical => {
                                (overlap_right - overlap_left).max(1) as f32
                            }
                        };
                        let position = match orientation {
                            SeamOrientation::Horizontal => {
                                (global_y.saturating_sub(overlap_top)) as f32 / span
                            }
                            SeamOrientation::Vertical => {
                                (global_x.saturating_sub(overlap_left)) as f32 / span
                            }
                        };
                        if new_image_is_dominant_side {
                            position.clamp(0.0, 1.0)
                        } else {
                            (1.0 - position).clamp(0.0, 1.0)
                        }
                    } else if psd_scan {
                        if candidate_valid { 1.0 } else { 0.0 }
                    } else {
                        panorama_detail_alpha(candidate_signed_distance)
                    };

                    let base_start = local_x as usize * 3;
                    base_row[base_start..base_start + 3].copy_from_slice(&base_pixel.0);
                    candidate_row[base_start..base_start + 3].copy_from_slice(&candidate_pixel.0);
                    blend_row[local_x as usize] = (detail_alpha * 255.0).round() as u8;
                    let low_frequency_alpha = if !candidate_valid {
                        0.0
                    } else if !panorama_valid {
                        1.0
                    } else {
                        match orientation {
                            SeamOrientation::Horizontal => {
                                let span = (overlap_bottom - overlap_top).max(1) as f32;
                                let position = (global_y.saturating_sub(overlap_top)) as f32 / span;
                                if new_image_is_dominant_side {
                                    position
                                } else {
                                    1.0 - position
                                }
                            }
                            SeamOrientation::Vertical => {
                                let span = (overlap_right - overlap_left).max(1) as f32;
                                let position =
                                    (global_x.saturating_sub(overlap_left)) as f32 / span;
                                if new_image_is_dominant_side {
                                    position
                                } else {
                                    1.0 - position
                                }
                            }
                        }
                    };
                    low_frequency_row[local_x as usize] =
                        (low_frequency_alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            },
        );

    let base = Rgb32FImage::from_raw(patch_width, patch_height, base_pixels)
        .expect("seam base patch dimensions must match");
    let candidate = Rgb32FImage::from_raw(patch_width, patch_height, candidate_pixels)
        .expect("seam candidate patch dimensions must match");
    let mask = GrayImage::from_raw(patch_width, patch_height, blend_mask.clone())
        .expect("seam mask dimensions must match");
    let low_frequency_mask = GrayImage::from_raw(patch_width, patch_height, low_frequency_mask)
        .expect("low-frequency seam mask dimensions must match");
    // Keep fine and middle-frequency Laplacian detail on one source side of the
    // path. Averaging those bands across the complete overlap softens strokes,
    // seals, foliage, and other structure whenever registration is not literally
    // pixel-identical. Only broad tonal bands may transition across the complete
    // overlap; visible detail transitions stay local to the optimal seam.
    let blended = multiband_blend(
        base,
        candidate,
        mask,
        Some(low_frequency_mask),
        PANORAMA_BLEND_BANDS,
        // A residual sub-pixel registration error must never be averaged into
        // the visible detail bands. Keep the source ownership hard for the
        // finest and middle frequencies; only the coarse tone pyramid is
        // allowed to feather across the seam.
        // PSD source scans are overlapping views of one plane.  Keeping all
        // high and middle bands hard-owned leaves a visible paper/brush
        // rectangle whenever the registration has a small residual.  Blend
        // the bands across the measured overlap for this path; ordinary
        // photographic panoramas retain hard detail ownership to avoid ghosting.
        !psd_scan,
    );

    let panorama_rgb_stride = out_width as usize * 3;
    let blended_pixels = blended.as_raw();
    panorama
        .as_mut()
        .par_chunks_mut(panorama_rgb_stride)
        .zip(panorama_mask.as_mut().par_chunks_mut(out_width as usize))
        .enumerate()
        .skip(patch_top as usize)
        .take(patch_height as usize)
        .for_each(|(global_y, (panorama_row, panorama_mask_row))| {
            let local_y = global_y - patch_top as usize;
            let blended_row =
                &blended_pixels[local_y * patch_rgb_stride..(local_y + 1) * patch_rgb_stride];
            let destination_start = patch_left as usize * 3;
            let destination_end = destination_start + patch_rgb_stride;
            panorama_row[destination_start..destination_end].copy_from_slice(blended_row);

            let blend_row =
                &blend_mask[local_y * patch_mask_stride..(local_y + 1) * patch_mask_stride];
            for (local_x, candidate_owns_pixel) in blend_row.iter().copied().enumerate() {
                let global_x = patch_left as usize + local_x;
                if panorama_mask_row[global_x] > 0 || candidate_owns_pixel > 0 {
                    panorama_mask_row[global_x] = 255;
                }
            }
        });
}
