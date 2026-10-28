//! Layer sampling primitives shared by both focus mosaic compositors.
//!
//! `Field` is the bilinear control-node grid behind every tone and residual
//! field; `LayerSampler` maps a layer position through its residual field and
//! pose to a prefiltered source sample and drops the black lower-left capture
//! border some handheld scan frames carry.

use super::super::stitching::Projection;
use super::super::stitching::map_target_to_source;
use crate::panorama_utils::shared::ImageInfo;
use image::Rgb;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use nalgebra::Point2;
use nalgebra::Point3;

use super::rgb_at;

pub(super) fn is_lower_left_black_capture_border(
    image: &Rgb32FImage,
    x: f64,
    y: f64,
    pixel: Rgb<f32>,
) -> bool {
    // Some handheld scan frames contain a genuinely black wedge outside the
    // photographed artwork in their lower-left corner. It is not source
    // content and must not become an owner merely because that frame extends
    // farther than its neighbours. Keep the rule deliberately local and very
    // dark: black robes, ink and furniture inside the image remain eligible.
    x < image.width() as f64 * 0.22
        && y > image.height() as f64 * 0.68
        && pixel[0].max(pixel[1]).max(pixel[2]) < 0.025
}

pub(in crate::panorama_utils) fn corrected_lower_left_capture_sample(
    _info: &ImageInfo,
    image: &Rgb32FImage,
    _source_x: f64,
    _source_y: f64,
    sample_x: f64,
    sample_y: f64,
    pixel: Rgb<f32>,
) -> Option<Rgb<f32>> {
    if is_lower_left_black_capture_border(image, sample_x, sample_y, pixel) {
        return None;
    }
    Some(pixel)
}

#[derive(Clone, Debug)]
pub(crate) struct Field<const N: usize> {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) step: f64,
    pub(crate) values: Vec<[f64; N]>,
}

impl<const N: usize> Field<N> {
    pub(crate) fn new(width: u32, height: u32, step: f64) -> Self {
        let width = (width as f64 / step).ceil() as usize + 1;
        let height = (height as f64 / step).ceil() as usize + 1;
        Self {
            width,
            height,
            step,
            values: vec![[0.0; N]; width * height],
        }
    }

    pub(crate) fn at(&self, x: f64, y: f64) -> [f64; N] {
        let gx = (x / self.step).clamp(0.0, self.width as f64 - 1.0);
        let gy = (y / self.step).clamp(0.0, self.height as f64 - 1.0);
        let x0 = gx as usize;
        let y0 = gy as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = gx - x0 as f64;
        let fy = gy - y0 as f64;
        std::array::from_fn(|c| {
            let a = self.values[y0 * self.width + x0][c] * (1.0 - fx)
                + self.values[y0 * self.width + x1][c] * fx;
            let b = self.values[y1 * self.width + x0][c] * (1.0 - fx)
                + self.values[y1 * self.width + x1][c] * fx;
            a * (1.0 - fy) + b * fy
        })
    }

    /// Construct a field with an explicit lower bound on the number of
    /// control nodes in each direction.  Residual_Warp uses this to retain a
    /// four-node support even for a very small overlap; the legacy mosaic
    /// field keeps the original geometry through [`Self::new`].
    pub(crate) fn new_with_min_nodes(width: u32, height: u32, step: f64, min_nodes: usize) -> Self {
        let mut field = Self::new(width, height, step);
        let min_nodes = min_nodes.max(1);
        if field.width < min_nodes || field.height < min_nodes {
            field.width = field.width.max(min_nodes);
            field.height = field.height.max(min_nodes);
            field.values.resize(field.width * field.height, [0.0; N]);
        }
        field
    }
}

pub(super) struct LayerSampler<'a> {
    pub(super) info: &'a ImageInfo,
    pub(super) source: &'a Rgb32FImage,
    pub(super) source_divisor: f64,
    pub(super) inverse: Matrix3<f64>,
    pub(super) native_mapping: Option<&'a (dyn Fn(Point3<f64>) -> Option<Point2<f64>> + Sync)>,
    pub(super) projection: Projection,
    pub(super) offset: (f64, f64),
    pub(super) left: u32,
    pub(super) top: u32,
    pub(super) scale: f64,
    pub(super) residual: Field<2>,
}

impl LayerSampler<'_> {
    pub(super) fn source_point(&self, x: f64, y: f64) -> Option<Point2<f64>> {
        let delta = self.residual.at(x / self.scale, y / self.scale);
        let target = Point3::new(
            self.left as f64 + x + delta[0] * self.scale - self.offset.0,
            self.top as f64 + y + delta[1] * self.scale - self.offset.1,
            1.0,
        );
        let source = if let Some(mapping) = self.native_mapping {
            mapping(target)?
        } else {
            map_target_to_source(&self.inverse, target, self.info, self.projection)?
        };
        if !self.info.source_is_supported(source)
            || source.x < 0.0
            || source.y < 0.0
            || source.x >= self.info.width as f64
            || source.y >= self.info.height as f64
        {
            return None;
        }
        Some(source)
    }

    pub(super) fn sample(&self, x: f64, y: f64) -> Option<Rgb<f32>> {
        let source = self.source_point(x, y)?;
        // Each prefilter level represents the centre of a 2x2 source block.
        // Retain the half-pixel offset to avoid a lens-dependent sampling shift.
        let sx = ((source.x + 0.5) / self.source_divisor - 0.5)
            .clamp(0.0, self.source.width() as f64 - 1.0);
        let sy = ((source.y + 0.5) / self.source_divisor - 0.5)
            .clamp(0.0, self.source.height() as f64 - 1.0);
        let pixel = rgb_at(self.source, sx, sy)?;
        corrected_lower_left_capture_sample(
            self.info,
            self.source,
            source.x,
            source.y,
            sx,
            sy,
            pixel,
        )
    }
}
