//! Several captures of one artwork in one stitch.
//!
//! A focus scan may combine several captures of one artwork (祁序1 and 祁序2
//! were photographed two days apart). The capture of the first source
//! composes the canvas: the canvas takes its gauge (see
//! `panorama_stitching::capture_gauge`) and its geometry, while the frames of
//! the other captures only fill what it leaves uncovered and lend their
//! detail to the repair (see `mosaic::detail_repair`).
//!
//! A capture is one camera's session: its frames, ordered by capture time,
//! with no pause longer than `SESSION_GAP` between them. Folders do not
//! decide it: 枫鹰雉鸡 was shot in one session and sorted afterwards into a
//! body, a head and a bottom folder, whose frames interleave in time, and
//! treating the bottom folder as a capture of its own left its twice as
//! sharp frames out of the composition. A source without a camera or a
//! capture time is its folder's capture.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use chrono::NaiveDateTime;

use super::shared::ImageInfo;

/// Longest pause between two frames of one session.
const SESSION_GAP_SECONDS: i64 = 30 * 60;

/// What a source's capture is told by: its camera and capture time, or its
/// folder.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum CaptureKey {
    Camera(String, String),
    Folder(String),
}

/// A source's camera and capture time, when its metadata holds both.
fn camera_and_time(info: &ImageInfo) -> Option<((String, String), NaiveDateTime)> {
    let bytes = crate::file_management::read_file_mapped(Path::new(&info.filename)).ok()?;
    let metadata = crate::exif_processing::read_exif_data_from_bytes(&info.filename, &bytes);
    let camera = (
        metadata.get("Make")?.clone(),
        metadata.get("Model")?.clone(),
    );
    let time = metadata.get("DateTimeOriginal")?;
    let time = ["%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S"]
        .iter()
        .find_map(|format| NaiveDateTime::parse_from_str(time.trim(), format).ok())?;
    Some((camera, time))
}

fn folder_of(info: &ImageInfo) -> String {
    Path::new(&info.filename)
        .parent()
        .map(|parent| parent.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// The capture of every source, by id: sessions numbered per camera in time
/// order, or the folder.
fn captures_of(images: &[&ImageInfo]) -> HashMap<usize, (CaptureKey, usize)> {
    let mut timed = Vec::new();
    let mut captures = HashMap::new();
    for info in images {
        match camera_and_time(info) {
            Some(((make, model), time)) => {
                timed.push((CaptureKey::Camera(make, model), time, info.id))
            }
            None => {
                captures.insert(info.id, (CaptureKey::Folder(folder_of(info)), 0));
            }
        }
    }
    timed.sort();
    let mut previous: Option<(CaptureKey, NaiveDateTime)> = None;
    let mut session = 0;
    for (camera, time, id) in timed {
        match &previous {
            Some((last_camera, last_time)) if *last_camera == camera => {
                if (time - *last_time).num_seconds() > SESSION_GAP_SECONDS {
                    session += 1;
                }
            }
            _ => session = 0,
        }
        previous = Some((camera.clone(), time));
        captures.insert(id, (camera, session));
    }
    captures
}

/// Ids of the frames of the composing capture, the capture of the source with
/// the lowest id, or `None` when every source belongs to one capture.
pub(crate) fn composing_capture_ids(images: &[&ImageInfo]) -> Option<HashSet<usize>> {
    let first = images.iter().min_by_key(|info| info.id)?;
    let captures = captures_of(images);
    let capture = captures.get(&first.id)?;
    let composing = images
        .iter()
        .filter(|info| captures.get(&info.id) == Some(capture))
        .map(|info| info.id)
        .collect::<HashSet<_>>();
    (composing.len() < images.len()).then_some(composing)
}

/// Whether the sources lie in more than one folder: a selection put together
/// from several shoots or several parts of one.
pub(crate) fn spans_several_folders(images: &[&ImageInfo]) -> bool {
    images
        .iter()
        .map(|info| folder_of(info))
        .collect::<HashSet<_>>()
        .len()
        > 1
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
