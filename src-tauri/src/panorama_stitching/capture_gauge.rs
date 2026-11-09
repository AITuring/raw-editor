//! The canvas gauge of a focus stack that combines several captures.
//!
//! A focus stack anchors its global poses to the first frame of its pose
//! graph. With several captures of one artwork in one selection the graph is
//! anchored to whichever frame its spanning tree starts from: stitched
//! together, 祁序2 composed the canvas at 0.61 canvas pixels per source pixel
//! where alone it composes at 0.90, so it came out at the other capture's
//! scale. Here the solved poses are re-expressed in the frame of the
//! composing capture (see `panorama_utils::captures`) that sits at the median
//! of that capture's canvas scales: the composing capture's typical frame
//! renders at one canvas pixel per source pixel, and the relative geometry
//! the solve found is unchanged. The order of the sources decides nothing:
//! anchored to the first photo listed, the fused canvas of 祁序1 came out a
//! third smaller than its photos, because that photo was shot from farther
//! away than the rest.

use nalgebra::Matrix3;
use std::collections::HashMap;
#[cfg(test)]
use std::collections::HashSet;

use super::ImageInfo;
use crate::panorama_utils::captures::{capture_members, composing_capture_ids};

/// Canvas pixels per source pixel at the centre of `image` under
/// `transform` (the smaller of the two axes), as the mosaic measures it.
fn canvas_scale(image: &ImageInfo, transform: &Matrix3<f64>) -> Option<f64> {
    let point = |x: f64, y: f64| {
        let projected = transform * nalgebra::Vector3::new(x, y, 1.0);
        (projected.x / projected.z, projected.y / projected.z)
    };
    let (cx, cy) = (f64::from(image.width) * 0.5, f64::from(image.height) * 0.5);
    let centre = point(cx, cy);
    let along_x = point(cx + 1.0, cy);
    let along_y = point(cx, cy + 1.0);
    let scale = (along_x.0 - centre.0)
        .hypot(along_x.1 - centre.1)
        .min((along_y.0 - centre.0).hypot(along_y.1 - centre.1));
    scale.is_finite().then_some(scale)
}

/// Canvas area per source pixel area at `(x, y)` of a frame under
/// `transform`, square-rooted.
fn local_scale(transform: &Matrix3<f64>, x: f64, y: f64) -> Option<f64> {
    let point = |x: f64, y: f64| {
        let projected = transform * nalgebra::Vector3::new(x, y, 1.0);
        (projected.x / projected.z, projected.y / projected.z)
    };
    let origin = point(x, y);
    let along_x = point(x + 1.0, y);
    let along_y = point(x, y + 1.0);
    let area = ((along_x.0 - origin.0) * (along_y.1 - origin.1)
        - (along_x.1 - origin.1) * (along_y.0 - origin.0))
        .abs();
    let scale = area.sqrt();
    (scale.is_finite() && scale > 0.0).then_some(scale)
}

/// How much more canvas a source pixel covers at one corner of the frame
/// than at another (corners inset by a tenth): one for a frame shot square
/// to the work, growing with the angle it was shot at.
fn perspective_spread(image: &ImageInfo, transform: &Matrix3<f64>) -> Option<f64> {
    let (width, height) = (f64::from(image.width), f64::from(image.height));
    let mut low = f64::INFINITY;
    let mut high = 0.0_f64;
    for (fx, fy) in [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)] {
        let scale = local_scale(transform, fx * width, fy * height)?;
        low = low.min(scale);
        high = high.max(scale);
    }
    Some(high / low)
}

/// Median over `ids` of the canvas pixels per source pixel at each frame's
/// centre (diagnostics).
#[cfg(test)]
pub(super) fn median_canvas_scale(
    images: &[ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
    ids: &HashSet<usize>,
) -> f64 {
    let mut scales = images
        .iter()
        .filter(|image| ids.contains(&image.id))
        .filter_map(|image| canvas_scale(image, homographies.get(&image.id)?))
        .collect::<Vec<_>>();
    if scales.is_empty() {
        return f64::NAN;
    }
    scales.sort_by(f64::total_cmp);
    scales[scales.len() / 2]
}

/// `homographies` re-anchored to the composing capture's median-scale frame
/// (ties broken by path), with that frame's index, or `None` when the
/// selection holds one capture or that frame has no invertible pose.
pub(super) fn composing_capture_gauge(
    images: &[ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
) -> Option<(usize, HashMap<usize, Matrix3<f64>>)> {
    let composing = composing_capture_ids(&images.iter().collect::<Vec<_>>())?;
    let mut scaled = images
        .iter()
        .enumerate()
        .filter(|(_, image)| composing.contains(&image.id))
        .filter_map(|(index, image)| {
            Some((canvas_scale(image, homographies.get(&image.id)?)?, index))
        })
        .collect::<Vec<_>>();
    scaled.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| images[a.1].filename.cmp(&images[b.1].filename))
    });
    let &(_, reference) = scaled.get(scaled.len().checked_sub(1)? / 2)?;
    let anchor = homographies.get(&images[reference].id)?.try_inverse()?;
    Some((reference, reanchored(homographies, &anchor)))
}

/// A frame is oblique when the spread of its canvas scale across its corners
/// (`perspective_spread`) exceeds its capture's median spread by more than
/// this: a corner of it covers a quarter more canvas per pixel than in the
/// capture's typical frame.
const OBLIQUE_SPREAD_EXCESS: f64 = 0.25;
/// A capture needs this many posed frames for its median to tell an oblique
/// frame from the rest.
const MIN_FRAMES_TO_TELL_AN_OBLIQUE_FRAME: usize = 8;

/// Indices into `images` of the frames of a scan shot at a much steeper angle
/// than the rest of their capture, with their perspective spread over the
/// capture's median. One such frame spans a wide stretch of the work under a
/// single strong perspective, and the solve bends the canvas to fit it:
/// DSC_3842 of 祁序1, shot obliquely across the work (spread 1.46 where its
/// capture's median is 1.12 and no other frame passes 1.25), turned the fused
/// canvas of 祁序 from the trapezoid its other frames compose into a near
/// rectangle a sixth smaller and doubled the inscription at 1:1.
pub(super) fn oblique_frames(
    images: &[ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
) -> Vec<(usize, f64)> {
    let index_of = images
        .iter()
        .enumerate()
        .map(|(index, image)| (image.id, index))
        .collect::<HashMap<_, _>>();
    let posed = images
        .iter()
        .filter(|image| homographies.contains_key(&image.id))
        .collect::<Vec<_>>();
    let mut oblique = Vec::new();
    for ids in capture_members(&posed) {
        let spreads = ids
            .iter()
            .filter_map(|id| {
                let index = *index_of.get(id)?;
                Some((
                    index,
                    perspective_spread(&images[index], homographies.get(id)?)?,
                ))
            })
            .collect::<Vec<_>>();
        if spreads.len() < MIN_FRAMES_TO_TELL_AN_OBLIQUE_FRAME {
            continue;
        }
        let mut sorted = spreads
            .iter()
            .map(|&(_, spread)| spread)
            .collect::<Vec<_>>();
        sorted.sort_by(f64::total_cmp);
        let median = sorted[sorted.len() / 2];
        #[cfg(test)]
        println!(
            "  - Capture of {} frames: median perspective spread {median:.3}, largest {}",
            spreads.len(),
            sorted
                .iter()
                .rev()
                .take(3)
                .map(|spread| format!("{spread:.3}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        oblique.extend(
            spreads
                .into_iter()
                .filter(|&(_, spread)| spread - median > OBLIQUE_SPREAD_EXCESS)
                .map(|(index, spread)| (index, spread / median)),
        );
    }
    oblique.sort_by_key(|&(index, _)| index);
    oblique
}

/// Every pose of `homographies` composed after `anchor` (the inverse of the
/// new reference's pose), normalised so its last entry is one.
fn reanchored(
    homographies: &HashMap<usize, Matrix3<f64>>,
    anchor: &Matrix3<f64>,
) -> HashMap<usize, Matrix3<f64>> {
    homographies
        .iter()
        .map(|(&id, homography)| {
            let mut pose = anchor * homography;
            let last = pose[(2, 2)];
            if last.abs() > 1e-12 {
                pose /= last;
            }
            (id, pose)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn similarity(scale: f64, angle: f64, tx: f64, ty: f64) -> Matrix3<f64> {
        let (sin, cos) = angle.sin_cos();
        Matrix3::new(
            scale * cos,
            -scale * sin,
            tx,
            scale * sin,
            scale * cos,
            ty,
            0.0,
            0.0,
            1.0,
        )
    }

    #[test]
    fn reanchoring_makes_the_new_reference_the_identity_and_keeps_relative_poses() {
        // Frame 0 anchors the solve; frame 2 belongs to a capture shot closer,
        // at 1.5 times the pixels per millimetre of artwork.
        let homographies = HashMap::from([
            (0, Matrix3::identity()),
            (1, similarity(1.0, 0.0, 900.0, 0.0)),
            (2, similarity(1.0 / 1.5, 0.02, 400.0, 300.0)),
            (3, similarity(1.0 / 1.5, 0.02, 1100.0, 310.0)),
        ]);
        let anchor = homographies[&2].try_inverse().expect("invertible");
        let result = reanchored(&homographies, &anchor);
        assert!((result[&2] - Matrix3::identity()).abs().max() < 1e-12);
        // Frame 0 now renders at the closer capture's scale.
        let scale = result[&0].fixed_view::<2, 2>(0, 0).determinant().sqrt();
        assert!((scale - 1.5).abs() < 1e-9, "scale {scale}");
        // Relative poses are unchanged.
        for (from, to) in [(0, 1), (2, 3), (1, 3)] {
            let before = homographies[&to].try_inverse().expect("invertible") * homographies[&from];
            let after = result[&to].try_inverse().expect("invertible") * result[&from];
            let before = before / before[(2, 2)];
            let after = after / after[(2, 2)];
            assert!((before - after).abs().max() < 1e-9, "{from}->{to}");
        }
    }
}
