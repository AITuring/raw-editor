//! The canvas gauge of a focus stack that combines several captures.
//!
//! A focus stack anchors its global poses to the first frame of its pose
//! graph. Stitched alone, 祁序1 and 祁序2 each anchored to their first source
//! or one of its focus brackets, so the first source rendered at one canvas
//! pixel per source pixel: the first photo is the output reference, as the
//! image-stack dialog tells the user. With several captures of one artwork in
//! one selection the graph is anchored to whichever frame its spanning tree
//! starts from: stitched together, 祁序2 composed the canvas at 0.61 canvas
//! pixels per source pixel where alone it composes at 0.90, so it came out
//! at the other capture's scale. Here the solved poses are re-expressed in
//! the frame of the first source, which belongs to the composing capture (see
//! `panorama_utils::captures`): the canvas takes the scale and orientation
//! that capture has when stitched alone, and the relative geometry the solve
//! found is unchanged.

use nalgebra::Matrix3;
use std::collections::HashMap;
#[cfg(test)]
use std::collections::HashSet;

use super::ImageInfo;
use crate::panorama_utils::captures::composing_capture_ids;

/// Median over `ids` of the canvas pixels per source pixel at each frame's
/// centre (the smaller of the two axes), as the mosaic measures it.
#[cfg(test)]
pub(super) fn median_canvas_scale(
    images: &[ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
    ids: &HashSet<usize>,
) -> f64 {
    let mut scales = images
        .iter()
        .filter(|image| ids.contains(&image.id))
        .filter_map(|image| {
            let transform = homographies.get(&image.id)?;
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
        })
        .collect::<Vec<_>>();
    if scales.is_empty() {
        return f64::NAN;
    }
    scales.sort_by(f64::total_cmp);
    scales[scales.len() / 2]
}

/// `homographies` re-anchored to the first source (the lowest id), with
/// that source's index, or `None` when the selection holds one capture or
/// the first source has no invertible pose.
pub(super) fn composing_capture_gauge(
    images: &[ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
) -> Option<(usize, HashMap<usize, Matrix3<f64>>)> {
    composing_capture_ids(&images.iter().collect::<Vec<_>>())?;
    let (reference, first) = images
        .iter()
        .enumerate()
        .min_by_key(|(_, image)| image.id)?;
    let anchor = homographies.get(&first.id)?.try_inverse()?;
    Some((reference, reanchored(homographies, &anchor)))
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
