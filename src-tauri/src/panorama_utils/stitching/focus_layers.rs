//! Rendered focus layers and Source_RAW sampling for focus stacks.
//!
//! Renders one source into the output plane through its focus warp and native
//! refinement, and keeps the inverse band lookup used by every sampler.

use super::super::coordinates::Projection;
use super::super::coordinates::map_target_to_source;
use super::super::render_geometry::focus_transformed_image_region;
use crate::panorama_stitching::FOCUS_FOREGROUND_LUMA_THRESHOLD;
use crate::panorama_utils::shared::FocusLayerWarp;
use crate::panorama_utils::shared::ImageInfo;
use image::GrayImage;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use nalgebra::Point2;
use nalgebra::Point3;
use rayon::prelude::*;

use super::{
    FocusSourceRefinement, get_high_quality_interpolated_pixel, source_sample_has_support,
};

pub(super) struct RenderedFocusLayer {
    pub(super) image: Rgb32FImage,
    pub(super) mask: GrayImage,
    pub(super) foreground_mask: GrayImage,
    // Pixels in a repeated long-edge silhouette may be switched between
    // aligned sources. Other detected foreground remains first-owner-wins.
    pub(super) relaxed_foreground_mask: GrayImage,
    pub(super) left: u32,
    pub(super) top: u32,
}

pub(super) struct RenderedFocusAnalysisLayer {
    pub(super) image: Rgb32FImage,
    pub(super) mask: GrayImage,
    pub(super) foreground_mask: GrayImage,
    pub(super) relaxed_foreground_mask: GrayImage,
}

pub(super) fn source_is_focus_foreground(image: &ImageInfo, source: Point2<f64>) -> bool {
    let Some((minimum_y, maximum_y)) = image.foreground_range else {
        return false;
    };
    let source_width = image.width().max(1) as f64;
    let source_height = image.height().max(1) as f64;
    if source.x < 0.0 || source.y < 0.0 || source.x >= source_width || source.y >= source_height {
        return false;
    }
    let source_y = source.y / source_height;
    if !(minimum_y..=maximum_y).contains(&source_y) {
        return false;
    }
    // Render metadata may intentionally omit the alignment pixels while
    // retaining its measured mask. Address that mask in normalized source
    // coordinates, so preview scaling cannot disable or move a depth band.
    let (alignment_width, alignment_height) = image.foreground_mask.as_ref().map_or_else(
        || image.alignment_image.dimensions(),
        |mask| mask.dimensions(),
    );
    let alignment_x = (source.x * f64::from(alignment_width) / source_width).round() as i32;
    let alignment_y = (source.y * f64::from(alignment_height) / source_height).round() as i32;
    if alignment_x < 0
        || alignment_y < 0
        || alignment_x >= alignment_width as i32
        || alignment_y >= alignment_height as i32
    {
        return false;
    }
    image.foreground_mask.as_ref().map_or_else(
        || {
            image
                .alignment_image
                .get_pixel(alignment_x as u32, alignment_y as u32)[0]
                >= FOCUS_FOREGROUND_LUMA_THRESHOLD
        },
        |mask| mask.get_pixel(alignment_x as u32, alignment_y as u32)[0] > 0,
    )
}

pub(super) type FocusBandInverse = (
    f64,
    f64,
    f64,
    f64,
    Matrix3<f64>,
    Matrix3<f64>,
    u8,
    bool,
    bool,
);

/// The shared inverse map used by rendering and the owner RAW reference.
#[derive(Clone)]
pub(crate) struct FocusSourceSampling {
    pub(super) image: std::sync::Arc<ImageInfo>,
    pub(super) projection: Projection,
    pub(super) global_inverse: Matrix3<f64>,
    pub(super) band_inverses: Vec<FocusBandInverse>,
}

impl std::fmt::Debug for FocusSourceSampling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FocusSourceSampling")
            .field("image_id", &self.image.id)
            .field("band_count", &self.band_inverses.len())
            .finish()
    }
}

impl FocusSourceSampling {
    pub(super) fn new(
        image: &ImageInfo,
        homography: &Matrix3<f64>,
        projection: Projection,
        focus_warp: Option<&FocusLayerWarp>,
    ) -> Self {
        let global_inverse = homography.try_inverse().unwrap_or_else(Matrix3::identity);
        let band_inverses = focus_warp
            .into_iter()
            .flat_map(|warp| warp.bands.iter())
            .filter_map(|band| {
                let homography = band.homographies.get(&image.id)?;
                let &(minimum_source_y, maximum_source_y) = band.source_ranges.get(&image.id)?;
                let (minimum_source_x, maximum_source_x) = band
                    .source_x_ranges
                    .get(&image.id)
                    .copied()
                    .unwrap_or((0.0, 1.0));
                let inverse = homography.try_inverse()?;
                let correction_inverse = (*homography * global_inverse).try_inverse()?;
                let axis =
                    if maximum_source_x - minimum_source_x < maximum_source_y - minimum_source_y {
                        0u8 // vertical edge: narrow source-x band
                    } else {
                        1u8 // horizontal/depth band: narrow source-y band
                    };
                Some((
                    minimum_source_x,
                    maximum_source_x,
                    minimum_source_y,
                    maximum_source_y,
                    inverse,
                    correction_inverse,
                    axis,
                    band.relax_foreground_seam,
                    band.foreground_only,
                ))
            })
            .collect::<Vec<_>>();

        Self {
            image: std::sync::Arc::new(ImageInfo {
                id: image.id,
                filename: image.filename.clone(),
                width: image.width,
                height: image.height,
                alignment_image: image.alignment_image.clone(),
                full_image: None,
                scale_factor: image.scale_factor,
                focal_length_35mm: image.focal_length_35mm,
                source_footprint: image.source_footprint,
                overview_reference: image.overview_reference,
                features: Vec::new(),
                top_features: Vec::new(),
                foreground_range: image.foreground_range,
                foreground_mask: image.foreground_mask.clone(),
                horizontal_edge_rows: Vec::new(),
                vertical_edge_columns: Vec::new(),
            }),
            projection,
            global_inverse,
            band_inverses,
        }
    }

    pub(crate) fn source_point(&self, target: Point3<f64>) -> Option<Point2<f64>> {
        self.mapping(target).0
    }

    pub(super) fn mapping(&self, target: Point3<f64>) -> (Option<Point2<f64>>, bool) {
        let image = &self.image;
        let global_inverse = self.global_inverse;
        let band_inverses = &self.band_inverses;
        let projection = self.projection;
        let global_source = map_target_to_source(&global_inverse, target, image, projection)
            .filter(|candidate| {
                image.source_is_supported(*candidate)
                    && candidate.x >= 0.0
                    && candidate.y >= 0.0
                    && candidate.x < image.width() as f64
                    && candidate.y < image.height() as f64
            });
        // A local model moves a depth layer relative to the paper. Do
        // not use the global inverse to decide whether that moved pixel
        // belongs to the band: doing so drops the local model exactly
        // along the displaced silhouette and creates a step back to the
        // paper transform. Instead, validate each local inverse in its
        // own source domain, then let the narrowest active band win.
        // This is content-agnostic; when no local bands exist the global
        // inverse remains the only mapping.
        let mut local_candidates = Vec::new();
        for &(
            minimum_source_x,
            maximum_source_x,
            minimum_source_y,
            maximum_source_y,
            ref band_inverse,
            ref correction_inverse,
            axis,
            relax_foreground_seam,
            foreground_only,
        ) in band_inverses
        {
            let Some(candidate) = map_target_to_source(band_inverse, target, image, projection)
            else {
                continue;
            };
            if !image.source_is_supported(candidate)
                || candidate.x < 0.0
                || candidate.y < 0.0
                || candidate.x >= image.width() as f64
                || candidate.y >= image.height() as f64
            {
                continue;
            }
            if foreground_only && !source_is_focus_foreground(image, candidate) {
                continue;
            }
            let normalized_x = candidate.x / image.width().max(1) as f64;
            let normalized_y = candidate.y / image.height().max(1) as f64;
            if normalized_x < minimum_source_x - 0.02
                || normalized_x > maximum_source_x + 0.02
                || normalized_y < minimum_source_y - 0.02
                || normalized_y > maximum_source_y + 0.02
            {
                continue;
            }
            let x_span = (maximum_source_x - minimum_source_x).max(1e-6);
            let y_span = (maximum_source_y - minimum_source_y).max(1e-6);
            let (span, normalized_axis, center_axis) = if x_span < y_span {
                (
                    x_span,
                    normalized_x,
                    (minimum_source_x + maximum_source_x) * 0.5,
                )
            } else {
                (
                    y_span,
                    normalized_y,
                    (minimum_source_y + maximum_source_y) * 0.5,
                )
            };
            local_candidates.push((
                span,
                normalized_axis,
                center_axis,
                axis,
                candidate,
                *correction_inverse,
                relax_foreground_seam,
                foreground_only,
            ));
        }
        let select_local_candidate = |axis: u8| {
            let narrowest_span = local_candidates
                .iter()
                .filter(|candidate| candidate.3 == axis)
                .map(|candidate| candidate.0)
                .fold(f64::INFINITY, f64::min);
            if !narrowest_span.is_finite() {
                return None;
            }
            let maximum_span = narrowest_span * 1.25;
            // Several generic/edge bands can overlap in source space.
            // Keep one candidate per axis: prefer the narrowest valid
            // band, then the band whose centre is closest to this
            // sample. Orthogonal bands are composed below rather than
            // allowing one axis to discard the other.
            local_candidates
                .iter()
                .filter(|candidate| candidate.3 == axis && candidate.0 <= maximum_span)
                .min_by(|left, right| {
                    left.0.total_cmp(&right.0).then_with(|| {
                        (left.1 - left.2)
                            .abs()
                            .total_cmp(&(right.1 - right.2).abs())
                    })
                })
        };
        let selected_vertical = select_local_candidate(0);
        let selected_horizontal = select_local_candidate(1);
        let local_strength_for =
            |candidate: &(f64, f64, f64, u8, Point2<f64>, Matrix3<f64>, bool, bool)| {
                (1.0 - (candidate.1 - candidate.2).abs() / (candidate.0 * 0.5)).clamp(0.0, 1.0)
            };
        let vertical_strength = selected_vertical.map(local_strength_for);
        let horizontal_strength = selected_horizontal.map(local_strength_for);
        let mut local_source = None;
        let mut local_strength = 0.0f64;
        let mut relax_foreground_seam = false;
        match (selected_vertical, selected_horizontal) {
            (Some(vertical), Some(horizontal)) => {
                // The stored correction is a world-space correction
                // on top of the global pose. Apply inverse horizontal
                // then inverse vertical correction before the global
                // inverse; this preserves both sides of a detected
                // corner/endpoint without introducing an averaged pose.
                let corrected_target = horizontal.5 * target;
                let corrected_target = vertical.5 * corrected_target;
                local_source =
                    map_target_to_source(&global_inverse, corrected_target, image, projection)
                        .filter(|candidate| {
                            candidate.x >= 0.0
                                && candidate.y >= 0.0
                                && candidate.x < image.width() as f64
                                && candidate.y < image.height() as f64
                        });
                local_strength = vertical_strength
                    .unwrap_or(0.0)
                    .min(horizontal_strength.unwrap_or(0.0));
                relax_foreground_seam = vertical.6 || horizontal.6;
            }
            (Some(vertical), None) => {
                local_source = Some(vertical.4);
                local_strength = vertical_strength.unwrap_or(0.0);
                relax_foreground_seam = vertical.6;
            }
            (None, Some(horizontal)) => {
                local_source = Some(horizontal.4);
                local_strength = horizontal_strength.unwrap_or(0.0);
                relax_foreground_seam = horizontal.6;
            }
            (None, None) => {}
        }
        if local_strength <= 0.0 {
            local_source = None;
            relax_foreground_seam = false;
        }
        // Blend each selected regional correction into the global
        // mapping at the edges of its source band. A hard switch is
        // itself visible as a seam, especially when the corrected
        // object is a rail or paper boundary. Each axis' influence
        // falls continuously to zero at its band boundary; when both
        // axes are active their composed correction uses the smaller
        // overlap weight. If the global inverse is unavailable, the
        // local mapping remains a valid fallback.
        let source = match (global_source, local_source) {
            (Some(global), Some(local)) => {
                let local_weight = local_strength.clamp(0.0, 1.0);
                Some(Point2::new(
                    global.x * (1.0 - local_weight) + local.x * local_weight,
                    global.y * (1.0 - local_weight) + local.y * local_weight,
                ))
            }
            (Some(global), None) => Some(global),
            (None, Some(local)) => Some(local),
            (None, None) => None,
        };
        (
            source.filter(|point| image.source_is_supported(*point)),
            relax_foreground_seam,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn render_focus_layer(
    image: &ImageInfo,
    source_image: &Rgb32FImage,
    homography: &Matrix3<f64>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
    offset_x: f64,
    offset_y: f64,
    out_width: u32,
    out_height: u32,
) -> RenderedFocusLayer {
    render_focus_layer_with_refinement(
        image,
        source_image,
        homography,
        projection,
        focus_warp,
        offset_x,
        offset_y,
        out_width,
        out_height,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn render_focus_layer_with_refinement(
    image: &ImageInfo,
    source_image: &Rgb32FImage,
    homography: &Matrix3<f64>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
    offset_x: f64,
    offset_y: f64,
    out_width: u32,
    out_height: u32,
    refinement: Option<&FocusSourceRefinement>,
    region: Option<(u32, u32, u32, u32)>,
) -> RenderedFocusLayer {
    let Some((left, right, top, bottom)) = region.or_else(|| {
        focus_transformed_image_region(
            image, homography, projection, focus_warp, offset_x, offset_y, out_width, out_height,
        )
    }) else {
        return RenderedFocusLayer {
            image: Rgb32FImage::new(0, 0),
            mask: GrayImage::new(0, 0),
            foreground_mask: GrayImage::new(0, 0),
            relaxed_foreground_mask: GrayImage::new(0, 0),
            left: 0,
            top: 0,
        };
    };
    let source_sampling = FocusSourceSampling::new(image, homography, projection, focus_warp);
    let layer_width = right - left + 1;
    let layer_height = bottom - top + 1;
    let mut pixels = vec![0.0f32; layer_width as usize * layer_height as usize * 3];
    let mut mask = vec![0u8; layer_width as usize * layer_height as usize];
    let mut foreground_mask = vec![0u8; layer_width as usize * layer_height as usize];
    let mut relaxed_foreground_mask = vec![0u8; layer_width as usize * layer_height as usize];

    pixels
        .par_chunks_mut(layer_width as usize * 3)
        .zip(mask.par_chunks_mut(layer_width as usize))
        .zip(foreground_mask.par_chunks_mut(layer_width as usize))
        .zip(relaxed_foreground_mask.par_chunks_mut(layer_width as usize))
        .enumerate()
        .for_each(
            |(y, (((row, mask_row), foreground_mask_row), relaxed_row))| {
                let global_y = top + y as u32;
                for local_x in 0..layer_width {
                    let global_x = left + local_x;
                    let canvas = Point2::new(global_x as f64, global_y as f64);
                    let canvas = refinement
                        .map_or(canvas, |refinement| refinement.inverse_canvas_point(canvas));
                    let target = Point3::new(canvas.x - offset_x, canvas.y - offset_y, 1.0);
                    let (source, relax_foreground_seam) = source_sampling.mapping(target);
                    let Some(source) = source else {
                        continue;
                    };
                    if source.x < 1.0
                        || source.y < 1.0
                        || source.x >= source_image.width() as f64 - 2.0
                        || source.y >= source_image.height() as f64 - 2.0
                        || !source_sample_has_support(source_image, source.x, source.y)
                    {
                        continue;
                    }
                    let pixel =
                        get_high_quality_interpolated_pixel(source_image, source.x, source.y);
                    // Requirement 3.6 forbids replacing an owned sample with a
                    // neighbouring "corrected" paper sample: the owner must
                    // identify the photograph and coordinate that supplied RGB.
                    let start = local_x as usize * 3;
                    row[start..start + 3].copy_from_slice(&pixel.0);
                    mask_row[local_x as usize] = 255;
                    // This mask controls focus ownership and duplicate suppression;
                    // it must stay geometric. Colour classification belongs only
                    // to the final canvas-tone protection below. Marking every
                    // black stroke or red seal here would make the first soft
                    // sample win forever and produce the reported ghosting.
                    if source_is_focus_foreground(image, source) {
                        foreground_mask_row[local_x as usize] = 255;
                        if relax_foreground_seam {
                            relaxed_row[local_x as usize] = 255;
                        }
                    }
                }
            },
        );

    RenderedFocusLayer {
        image: Rgb32FImage::from_raw(layer_width, layer_height, pixels)
            .expect("focus layer buffer dimensions must match"),
        mask: GrayImage::from_raw(layer_width, layer_height, mask)
            .expect("focus layer mask dimensions must match"),
        foreground_mask: GrayImage::from_raw(layer_width, layer_height, foreground_mask)
            .expect("focus foreground mask buffer dimensions must match"),
        relaxed_foreground_mask: GrayImage::from_raw(
            layer_width,
            layer_height,
            relaxed_foreground_mask,
        )
        .expect("focus relaxed foreground mask dimensions must match"),
        left,
        top,
    }
}
