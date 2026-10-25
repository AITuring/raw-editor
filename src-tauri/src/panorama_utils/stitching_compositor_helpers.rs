//! Small, deterministic canvas helpers used by the ownership compositor.
//!
//! These operations only inspect the rendered coverage mask and crop the
//! corresponding image. Keeping them outside the ownership sampling loop
//! makes the compositor's finishing policy easier to reason about and test.

use image::{GrayImage, Rgb32FImage};

/// Return the largest all-valid rectangle in a coverage mask and crop the
/// image to it. The histogram-stack implementation is linear in the number of
/// mask pixels and handles projective corner wedges without scanning every
/// candidate rectangle.
pub(crate) fn crop_to_valid_rectangle(image: Rgb32FImage, mask: &GrayImage) -> Rgb32FImage {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || mask.dimensions() != (width, height) {
        return image;
    }

    let mut heights = vec![0usize; width as usize];
    let mut stack = Vec::with_capacity(width as usize + 1);
    let mut best_area = 0usize;
    let mut best_left = 0usize;
    let mut best_top = 0usize;
    let mut best_width = width as usize;
    let mut best_height = height as usize;

    for y in 0..height as usize {
        for (x, height_value) in heights.iter_mut().enumerate() {
            *height_value = if mask.get_pixel(x as u32, y as u32)[0] > 0 {
                height_value.saturating_add(1)
            } else {
                0
            };
        }

        stack.clear();
        for x in 0..=width as usize {
            let current_height = if x < width as usize { heights[x] } else { 0 };
            while let Some(&bar_index) = stack.last() {
                if heights[bar_index] <= current_height {
                    break;
                }
                stack.pop();
                let left = stack.last().map_or(0, |&index| index + 1);
                let rectangle_width = x - left;
                let rectangle_height = heights[bar_index];
                let area = rectangle_width * rectangle_height;
                if area > best_area {
                    best_area = area;
                    best_left = left;
                    best_top = y + 1 - rectangle_height;
                    best_width = rectangle_width;
                    best_height = rectangle_height;
                }
            }
            stack.push(x);
        }
    }

    if best_area == 0 {
        return image;
    }

    if best_left == 0
        && best_top == 0
        && best_width == width as usize
        && best_height == height as usize
    {
        return image;
    }

    image::imageops::crop_imm(
        &image,
        best_left as u32,
        best_top as u32,
        best_width as u32,
        best_height as u32,
    )
    .to_image()
}

/// Return the coverage mask bounding box and the fraction of that box covered
/// by valid pixels. A projective tile union can have transparent corner wedges
/// even though its outer bounds are useful; the coverage lets callers choose a
/// source-union crop only when those wedges are negligible.
pub(crate) fn valid_mask_bounds_and_coverage(mask: &GrayImage) -> ((u32, u32, u32, u32), f32) {
    let (width, height) = mask.dimensions();
    if width == 0 || height == 0 {
        return ((0, 0, 0, 0), 0.0);
    }
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut valid = 0u64;
    for y in 0..height {
        for x in 0..width {
            if mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            valid += 1;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if valid == 0 {
        return ((0, 0, 0, 0), 0.0);
    }
    let bbox_width = u64::from(max_x - min_x + 1);
    let bbox_height = u64::from(max_y - min_y + 1);
    let coverage = valid as f32 / (bbox_width * bbox_height).max(1) as f32;
    ((min_x, min_y, max_x, max_y), coverage.clamp(0.0, 1.0))
}

/// Crop an image to the axis-aligned bounding box of all covered pixels.
/// Uncovered projective corners remain transparent inside that union box.
pub(crate) fn crop_to_valid_bounds(image: Rgb32FImage, mask: &GrayImage) -> Rgb32FImage {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || mask.dimensions() != (width, height) {
        return image;
    }
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut covered = false;
    for y in 0..height {
        for x in 0..width {
            if mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            covered = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if !covered {
        return image;
    }
    let crop_width = max_x - min_x + 1;
    let crop_height = max_y - min_y + 1;
    if min_x == 0 && min_y == 0 && crop_width == width && crop_height == height {
        return image;
    }
    image::imageops::crop_imm(&image, min_x, min_y, crop_width, crop_height).to_image()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgb};

    #[test]
    fn valid_bounds_crop_keeps_the_union_bbox() {
        let mut image = Rgb32FImage::new(4, 3);
        for (index, pixel) in image.pixels_mut().enumerate() {
            *pixel = Rgb([index as f32, 0.0, 0.0]);
        }
        let mut mask = GrayImage::new(4, 3);
        mask.put_pixel(1, 1, Luma([255]));
        mask.put_pixel(2, 1, Luma([255]));

        let cropped = crop_to_valid_bounds(image, &mask);
        assert_eq!(cropped.dimensions(), (2, 1));
        assert_eq!(cropped.get_pixel(0, 0)[0], 5.0);
        assert_eq!(cropped.get_pixel(1, 0)[0], 6.0);
    }

    #[test]
    fn coverage_reports_valid_fraction_inside_the_bbox() {
        let mut mask = GrayImage::new(4, 3);
        mask.put_pixel(1, 1, Luma([255]));
        mask.put_pixel(2, 1, Luma([255]));
        mask.put_pixel(1, 2, Luma([255]));

        let (bounds, coverage) = valid_mask_bounds_and_coverage(&mask);
        assert_eq!(bounds, (1, 1, 2, 2));
        assert!((coverage - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn valid_rectangle_crop_uses_the_largest_contiguous_region() {
        let image = Rgb32FImage::new(5, 4);
        let mut mask = GrayImage::new(5, 4);
        for y in 0..3 {
            for x in 1..4 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        mask.put_pixel(0, 0, Luma([255]));
        let cropped = crop_to_valid_rectangle(image, &mask);
        assert_eq!(cropped.dimensions(), (3, 3));
    }
}
