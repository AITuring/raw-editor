//! Grow and feather post-processing for sub-masks.
//!
//! Separable grayscale dilation and erosion and the Gaussian feather plan whose
//! halo bounds tiled generation.

use image::GrayImage;

pub(super) fn grayscale_dilate(image: &GrayImage, k: u8) -> GrayImage {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return image.clone();
    }
    let w = width as usize;
    let h = height as usize;
    let r = k as i32;
    let src = image.as_raw();

    let mut temp = vec![0u8; w * h];
    let mut out = vec![0u8; w * h];

    for y in 0..h {
        let row_offset = y * w;
        for x in 0..w {
            let mut max_val = 0;
            let start = (x as i32 - r).max(0) as usize;
            let end = (x as i32 + r).min((w - 1) as i32) as usize;
            for xi in start..=end {
                max_val = max_val.max(src[row_offset + xi]);
            }
            temp[row_offset + x] = max_val;
        }
    }

    for x in 0..w {
        for y in 0..h {
            let mut max_val = 0;
            let start = (y as i32 - r).max(0) as usize;
            let end = (y as i32 + r).min((h - 1) as i32) as usize;
            for yi in start..=end {
                max_val = max_val.max(temp[yi * w + x]);
            }
            out[y * w + x] = max_val;
        }
    }

    GrayImage::from_raw(width, height, out).unwrap()
}

pub(super) fn grayscale_erode(image: &GrayImage, k: u8) -> GrayImage {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return image.clone();
    }
    let w = width as usize;
    let h = height as usize;
    let r = k as i32;
    let src = image.as_raw();

    let mut temp = vec![0u8; w * h];
    let mut out = vec![0u8; w * h];

    for y in 0..h {
        let row_offset = y * w;
        for x in 0..w {
            let mut min_val = 255;
            let start = (x as i32 - r).max(0) as usize;
            let end = (x as i32 + r).min((w - 1) as i32) as usize;
            for xi in start..=end {
                min_val = min_val.min(src[row_offset + xi]);
            }
            temp[row_offset + x] = min_val;
        }
    }

    for x in 0..w {
        for y in 0..h {
            let mut min_val = 255;
            let start = (y as i32 - r).max(0) as usize;
            let end = (y as i32 + r).min((h - 1) as i32) as usize;
            for yi in start..=end {
                min_val = min_val.min(temp[yi * w + x]);
            }
            out[y * w + x] = min_val;
        }
    }

    GrayImage::from_raw(width, height, out).unwrap()
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct GrowFeatherPlan {
    pub(super) grow_amount: u8,
    pub(super) dilate: bool,
    pub(super) feather_sigma: f32,
    pub(super) feather_radius: u32,
}

impl GrowFeatherPlan {
    pub(super) fn new(grow: f32, feather: f32, width: u32, height: u32) -> Self {
        let base_dimension = width.min(height) as f32;
        let grow_pixels = if grow.abs() > 0.01 {
            const MAX_GROW_PERCENTAGE: f32 = 0.01;
            (grow / 100.0) * base_dimension * MAX_GROW_PERCENTAGE
        } else {
            0.0
        };
        let grow_amount = grow_pixels.abs().round() as u8;

        let feather_sigma = if feather > 0.0 {
            const MAX_FEATHER_SIGMA_PERCENTAGE: f32 = 0.005;
            let sigma = (feather / 100.0) * base_dimension * MAX_FEATHER_SIGMA_PERCENTAGE;
            if sigma > 0.01 { sigma } else { 0.0 }
        } else {
            0.0
        };

        Self {
            grow_amount,
            dilate: grow_pixels > 0.0,
            feather_sigma,
            // imageproc::gaussian_blur_f32 constructs a finite kernel with this radius.
            feather_radius: (2.0 * feather_sigma).ceil() as u32,
        }
    }

    pub(super) fn halo(self) -> u32 {
        u32::from(self.grow_amount).saturating_add(self.feather_radius)
    }

    pub(super) fn has_effect(self) -> bool {
        self.grow_amount > 0 || self.feather_sigma > 0.0
    }

    pub(super) fn apply(self, mask: &mut GrayImage) {
        if self.grow_amount > 0 {
            if self.dilate {
                *mask = grayscale_dilate(mask, self.grow_amount);
            } else {
                *mask = grayscale_erode(mask, self.grow_amount);
            }
        }

        if self.feather_sigma > 0.0 {
            *mask = imageproc::filter::gaussian_blur_f32(mask, self.feather_sigma);
        }
    }
}

pub(super) fn apply_grow_and_feather(
    mask: &mut GrayImage,
    grow: f32,
    feather: f32,
    width: u32,
    height: u32,
) {
    GrowFeatherPlan::new(grow, feather, width, height).apply(mask);
}
