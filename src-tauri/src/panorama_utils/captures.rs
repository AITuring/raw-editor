//! Several captures of one artwork in one stitch.
//!
//! A focus scan may combine several captures of one artwork, one folder each
//! (祁序1 and 祁序2 were photographed on different occasions). The capture of
//! the first source composes the canvas: the canvas takes its gauge (see
//! `panorama_stitching::capture_gauge`), its tone and its geometry, while the
//! frames of the other captures only fill what it leaves uncovered and lend
//! their detail to the repair (see `mosaic::detail_repair`).

use std::collections::HashSet;
use std::path::Path;

use super::shared::ImageInfo;

/// The capture a source belongs to: the folder it lies in.
fn capture_of(info: &ImageInfo) -> Option<&Path> {
    Path::new(&info.filename).parent()
}

/// Ids of the frames of the composing capture, the capture of the source with
/// the lowest id, or `None` when every source belongs to one capture.
pub(crate) fn composing_capture_ids(images: &[&ImageInfo]) -> Option<HashSet<usize>> {
    let first = images.iter().min_by_key(|info| info.id)?;
    let capture = capture_of(first);
    let composing = images
        .iter()
        .filter(|info| capture_of(info) == capture)
        .map(|info| info.id)
        .collect::<HashSet<_>>();
    (composing.len() < images.len()).then_some(composing)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(id: usize, filename: &str) -> ImageInfo {
        ImageInfo {
            id,
            filename: filename.to_string(),
            width: 8,
            height: 8,
            alignment_image: image::GrayImage::new(1, 1),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm: None,
            source_footprint: None,
            overview_reference: false,
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: Vec::new(),
            vertical_edge_columns: Vec::new(),
        }
    }

    #[test]
    fn the_capture_of_the_first_source_composes() {
        let images = [
            info(2, "/scans/second/DSC_0001.NEF"),
            info(0, "/scans/first/DSC_0101.NEF"),
            info(1, "/scans/first/DSC_0102.NEF"),
            info(3, "/scans/second/DSC_0002.NEF"),
        ];
        let refs = images.iter().collect::<Vec<_>>();
        assert_eq!(composing_capture_ids(&refs), Some(HashSet::from([0, 1])));
    }

    #[test]
    fn one_capture_has_no_composing_capture() {
        let images = [
            info(0, "/scans/first/DSC_0101.NEF"),
            info(1, "/scans/first/DSC_0102.NEF"),
        ];
        let refs = images.iter().collect::<Vec<_>>();
        assert_eq!(composing_capture_ids(&refs), None);
    }
}
