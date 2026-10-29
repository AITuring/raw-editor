//! Source sampling for the geometry warp.
//!
//! Borrowed or converted pixel sources, the per-warp sample context, bilinear
//! interpolation (optionally with lens TCA) and the buffer budget of a
//! materialised warp.

use image::DynamicImage;
use image::Rgba32FImage;
use std::mem::size_of;

#[inline(always)]
pub(super) fn interpolate_pixel(
    src_raw: &[f32],
    src_width: usize,
    src_height: usize,
    x: f32,
    y: f32,
    pixel_out: &mut [f32],
) {
    if x.is_nan()
        || y.is_nan()
        || x < 0.0
        || y < 0.0
        || x >= (src_width as f32 - 1.0)
        || y >= (src_height as f32 - 1.0)
    {
        return;
    }

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;

    let wx = x - x0 as f32;
    let wy = y - y0 as f32;
    let one_minus_wx = 1.0 - wx;
    let one_minus_wy = 1.0 - wy;

    let stride = src_width * 3;
    let idx_row0 = y0 * stride;
    let idx_row1 = idx_row0 + stride;
    let idx_p00 = idx_row0 + x0 * 3;

    unsafe {
        let p00 = src_raw.get_unchecked(idx_p00..idx_p00 + 3);
        let p10 = src_raw.get_unchecked(idx_p00 + 3..idx_p00 + 6);
        let p01 = src_raw.get_unchecked(idx_row1 + x0 * 3..idx_row1 + x0 * 3 + 3);
        let p11 = src_raw.get_unchecked(idx_row1 + x0 * 3 + 3..idx_row1 + x0 * 3 + 6);

        let top_r = p00[0] * one_minus_wx + p10[0] * wx;
        let top_g = p00[1] * one_minus_wx + p10[1] * wx;
        let top_b = p00[2] * one_minus_wx + p10[2] * wx;

        let bot_r = p01[0] * one_minus_wx + p11[0] * wx;
        let bot_g = p01[1] * one_minus_wx + p11[1] * wx;
        let bot_b = p01[2] * one_minus_wx + p11[2] * wx;

        pixel_out[0] = top_r * one_minus_wy + bot_r * wy;
        pixel_out[1] = top_g * one_minus_wy + bot_g * wy;
        pixel_out[2] = top_b * one_minus_wy + bot_b * wy;
    }
}

pub(super) enum GeometryPixelSource<'a> {
    Luma8(&'a [u8]),
    LumaA8(&'a [u8]),
    Rgb8(&'a [u8]),
    Rgba8(&'a [u8]),
    Luma16(&'a [u16]),
    LumaA16(&'a [u16]),
    Rgb16(&'a [u16]),
    Rgba16(&'a [u16]),
    Rgb32F(&'a [f32]),
    Rgba32F(&'a [f32]),
    ConvertedRgba32F(Rgba32FImage),
}

impl<'a> GeometryPixelSource<'a> {
    pub(super) fn new(image: &'a DynamicImage) -> Self {
        match image {
            DynamicImage::ImageRgb32F(source) => Self::Rgb32F(source.as_raw()),
            DynamicImage::ImageRgba32F(source) => Self::Rgba32F(source.as_raw()),
            _ => Self::ConvertedRgba32F(image.to_rgba32f()),
        }
    }

    pub(super) fn borrowed(image: &'a DynamicImage) -> Self {
        match image {
            DynamicImage::ImageLuma8(source) => Self::Luma8(source.as_raw()),
            DynamicImage::ImageLumaA8(source) => Self::LumaA8(source.as_raw()),
            DynamicImage::ImageRgb8(source) => Self::Rgb8(source.as_raw()),
            DynamicImage::ImageRgba8(source) => Self::Rgba8(source.as_raw()),
            DynamicImage::ImageLuma16(source) => Self::Luma16(source.as_raw()),
            DynamicImage::ImageLumaA16(source) => Self::LumaA16(source.as_raw()),
            DynamicImage::ImageRgb16(source) => Self::Rgb16(source.as_raw()),
            DynamicImage::ImageRgba16(source) => Self::Rgba16(source.as_raw()),
            DynamicImage::ImageRgb32F(source) => Self::Rgb32F(source.as_raw()),
            DynamicImage::ImageRgba32F(source) => Self::Rgba32F(source.as_raw()),
            _ => Self::ConvertedRgba32F(image.to_rgba32f()),
        }
    }

    pub(super) fn sample_data(&self) -> GeometrySampleData<'_> {
        match self {
            Self::Luma8(raw) => GeometrySampleData::Luma8(raw),
            Self::LumaA8(raw) => GeometrySampleData::LumaA8(raw),
            Self::Rgb8(raw) => GeometrySampleData::Rgb8(raw),
            Self::Rgba8(raw) => GeometrySampleData::Rgba8(raw),
            Self::Luma16(raw) => GeometrySampleData::Luma16(raw),
            Self::LumaA16(raw) => GeometrySampleData::LumaA16(raw),
            Self::Rgb16(raw) => GeometrySampleData::Rgb16(raw),
            Self::Rgba16(raw) => GeometrySampleData::Rgba16(raw),
            Self::Rgb32F(raw) => GeometrySampleData::Rgb32F(raw),
            Self::Rgba32F(raw) => GeometrySampleData::Rgba32F(raw),
            Self::ConvertedRgba32F(source) => GeometrySampleData::Rgba32F(source.as_raw()),
        }
    }

    pub(super) fn conversion_bytes(&self) -> usize {
        match self {
            Self::ConvertedRgba32F(source) => source.as_raw().len() * size_of::<f32>(),
            _ => 0,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum GeometrySampleData<'a> {
    Luma8(&'a [u8]),
    LumaA8(&'a [u8]),
    Rgb8(&'a [u8]),
    Rgba8(&'a [u8]),
    Luma16(&'a [u16]),
    LumaA16(&'a [u16]),
    Rgb16(&'a [u16]),
    Rgba16(&'a [u16]),
    Rgb32F(&'a [f32]),
    Rgba32F(&'a [f32]),
}

impl GeometrySampleData<'_> {
    #[inline(always)]
    pub(super) fn sample(self, pixel: usize, channel: usize) -> f32 {
        debug_assert!(channel < 4);
        match self {
            Self::Luma8(raw) => {
                if channel == 3 {
                    1.0
                } else {
                    f32::from(raw[pixel]) / 255.0
                }
            }
            Self::LumaA8(raw) => {
                let component = if channel == 3 { 1 } else { 0 };
                f32::from(raw[pixel * 2 + component]) / 255.0
            }
            Self::Rgb8(raw) => {
                if channel == 3 {
                    1.0
                } else {
                    f32::from(raw[pixel * 3 + channel]) / 255.0
                }
            }
            Self::Rgba8(raw) => f32::from(raw[pixel * 4 + channel]) / 255.0,
            Self::Luma16(raw) => {
                if channel == 3 {
                    1.0
                } else {
                    f32::from(raw[pixel]) / 65_535.0
                }
            }
            Self::LumaA16(raw) => {
                let component = if channel == 3 { 1 } else { 0 };
                f32::from(raw[pixel * 2 + component]) / 65_535.0
            }
            Self::Rgb16(raw) => {
                if channel == 3 {
                    1.0
                } else {
                    f32::from(raw[pixel * 3 + channel]) / 65_535.0
                }
            }
            Self::Rgba16(raw) => f32::from(raw[pixel * 4 + channel]) / 65_535.0,
            Self::Rgb32F(raw) => {
                if channel == 3 {
                    1.0
                } else {
                    raw[pixel * 3 + channel]
                }
            }
            Self::Rgba32F(raw) => raw[pixel * 4 + channel],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GeometryWarpBufferPlan {
    pub(super) output_samples: usize,
    pub(super) output_rgba32f_bytes: usize,
    pub(super) source_conversion_bytes: usize,
}

impl GeometryWarpBufferPlan {
    pub(super) fn new(
        width: u32,
        height: u32,
        source_conversion_bytes: usize,
    ) -> Option<GeometryWarpBufferPlan> {
        let pixels = (width as usize).checked_mul(height as usize)?;
        let output_samples = pixels.checked_mul(4)?;
        Some(Self {
            output_samples,
            output_rgba32f_bytes: output_samples.checked_mul(size_of::<f32>())?,
            source_conversion_bytes,
        })
    }

    pub(super) fn transient_bytes(&self) -> usize {
        self.output_rgba32f_bytes + self.source_conversion_bytes
    }
}

#[derive(Clone, Copy)]
pub(super) struct GeometrySampleContext<'a> {
    pub(super) source: GeometrySampleData<'a>,
    pub(super) src_width: usize,
    pub(super) src_height: usize,
    pub(super) cx: f32,
    pub(super) cy: f32,
}

impl GeometrySampleContext<'_> {
    #[inline(always)]
    pub(super) fn contains_bilinear_sample(&self, x: f32, y: f32) -> bool {
        x.is_finite()
            && y.is_finite()
            && x >= 0.0
            && y >= 0.0
            && x < self.src_width as f32 - 1.0
            && y < self.src_height as f32 - 1.0
    }

    #[inline(always)]
    pub(super) fn sample_channel_clamped(&self, x: f32, y: f32, channel: usize) -> f32 {
        if !x.is_finite() || !y.is_finite() {
            return 0.0;
        }
        let x = x.clamp(0.0, self.src_width as f32 - 1.0);
        let y = y.clamp(0.0, self.src_height as f32 - 1.0);
        let mut x0 = x.floor() as usize;
        let mut y0 = y.floor() as usize;
        if x0 >= self.src_width - 1 {
            x0 = self.src_width.saturating_sub(2);
        }
        if y0 >= self.src_height - 1 {
            y0 = self.src_height.saturating_sub(2);
        }

        let wx = x - x0 as f32;
        let wy = y - y0 as f32;
        let one_minus_wx = 1.0 - wx;
        let one_minus_wy = 1.0 - wy;
        let row0 = y0 * self.src_width;
        let row1 = row0 + self.src_width;
        let p00 = self.source.sample(row0 + x0, channel);
        let p10 = self.source.sample(row0 + x0 + 1, channel);
        let p01 = self.source.sample(row1 + x0, channel);
        let p11 = self.source.sample(row1 + x0 + 1, channel);
        let top = p00 * one_minus_wx + p10 * wx;
        let bottom = p01 * one_minus_wx + p11 * wx;
        top * one_minus_wy + bottom * wy
    }
}

#[inline(always)]
pub(super) fn interpolate_geometry_pixel(
    source: &GeometrySampleContext<'_>,
    x: f32,
    y: f32,
    pixel_out: &mut [f32],
) {
    if !source.contains_bilinear_sample(x, y) {
        return;
    }
    for (channel, output) in pixel_out.iter_mut().enumerate().take(4) {
        *output = source.sample_channel_clamped(x, y, channel);
    }
}

#[inline(always)]
pub(super) fn interpolate_pixel_with_tca(
    source: &GeometrySampleContext<'_>,
    base_x: f32,
    base_y: f32,
    vr: f32,
    vb: f32,
    pixel_out: &mut [f32],
) {
    let cx = source.cx;
    let cy = source.cy;
    let gx = base_x;
    let gy = base_y;

    let rx = cx + (base_x - cx) * vr;
    let ry = cy + (base_y - cy) * vr;

    let bx = cx + (base_x - cx) * vb;
    let by = cy + (base_y - cy) * vb;

    if !source.contains_bilinear_sample(base_x, base_y) {
        return;
    }

    pixel_out[0] = source.sample_channel_clamped(rx, ry, 0);
    pixel_out[1] = source.sample_channel_clamped(gx, gy, 1);
    pixel_out[2] = source.sample_channel_clamped(bx, by, 2);
    pixel_out[3] = source.sample_channel_clamped(gx, gy, 3);
}
