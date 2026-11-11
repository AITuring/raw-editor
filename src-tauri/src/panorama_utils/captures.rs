//! Several captures of one artwork in one stitch.
//!
//! A focus scan may combine several captures of one artwork (祁序1 and 祁序2
//! were photographed two days apart). The largest capture composes the
//! canvas: the canvas takes its gauge (see
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
fn camera_and_time(path: &str) -> Option<((String, String), NaiveDateTime)> {
    let bytes = crate::file_management::read_file_mapped(Path::new(path)).ok()?;
    let metadata = crate::exif_processing::read_exif_data_from_bytes(path, &bytes);
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

fn folder_of(path: &str) -> String {
    Path::new(path)
        .parent()
        .map(|parent| parent.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// The capture of every source, by id: sessions numbered per camera in time
/// order, or the folder.
fn captures_of(sources: &[(usize, &str)]) -> HashMap<usize, (CaptureKey, usize)> {
    let mut timed = Vec::new();
    let mut captures = HashMap::new();
    for &(id, path) in sources {
        match camera_and_time(path) {
            Some(((make, model), time)) => timed.push((CaptureKey::Camera(make, model), time, id)),
            None => {
                captures.insert(id, (CaptureKey::Folder(folder_of(path)), 0));
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

/// Ids of the frames of the composing capture, or `None` when every source
/// belongs to one capture. The composing capture is the one with the most
/// frames, ties going to the capture whose earliest frame sorts first by
/// path; the order the sources were listed in decides nothing, so the
/// image-stack dialog composes the same canvas from any order of the same
/// photos.
pub(crate) fn composing_capture_ids(images: &[&ImageInfo]) -> Option<HashSet<usize>> {
    composing_capture(
        &images
            .iter()
            .map(|info| (info.id, info.filename.as_str()))
            .collect::<Vec<_>>(),
    )
}

/// Ids of the frames of every capture, each capture's ids sorted.
pub(crate) fn capture_members(images: &[&ImageInfo]) -> Vec<Vec<usize>> {
    let captures = captures_of(
        &images
            .iter()
            .map(|info| (info.id, info.filename.as_str()))
            .collect::<Vec<_>>(),
    );
    let mut members = HashMap::<&(CaptureKey, usize), Vec<usize>>::new();
    for (id, capture) in &captures {
        members.entry(capture).or_default().push(*id);
    }
    let mut members = members.into_values().collect::<Vec<_>>();
    for ids in &mut members {
        ids.sort_unstable();
    }
    members.sort();
    members
}

/// [`composing_capture_ids`] over `(id, path)` pairs.
fn composing_capture(sources: &[(usize, &str)]) -> Option<HashSet<usize>> {
    let captures = captures_of(sources);
    let mut members = HashMap::<&(CaptureKey, usize), (usize, &str)>::new();
    for &(id, path) in sources {
        let Some(capture) = captures.get(&id) else {
            continue;
        };
        let entry = members.entry(capture).or_insert((0, path));
        entry.0 += 1;
        if path < entry.1 {
            entry.1 = path;
        }
    }
    if members.len() < 2 {
        return None;
    }
    let composing = members
        .iter()
        .max_by(|a, b| a.1.0.cmp(&b.1.0).then_with(|| b.1.1.cmp(a.1.1)))
        .map(|(capture, _)| (*capture).clone())?;
    Some(
        sources
            .iter()
            .filter(|(id, _)| captures.get(id) == Some(&composing))
            .map(|&(id, _)| id)
            .collect(),
    )
}

/// `paths` in the order a focus stack processes them: the composing
/// capture's photos first, then the rest, each part by path. The pose solve
/// grows from the first photos, and grown from 祁序1 it bent the fused canvas
/// of 祁序 into a steeper trapezoid than grown from 祁序2, the capture that
/// composes it.
pub(crate) fn processing_order(paths: &mut [String]) {
    paths.sort();
    let composing = composing_capture(
        &paths
            .iter()
            .enumerate()
            .map(|(index, path)| (index, path.as_str()))
            .collect::<Vec<_>>(),
    );
    if let Some(composing) = composing {
        let mut keyed = paths
            .iter()
            .enumerate()
            .map(|(index, path)| (!composing.contains(&index), path.clone()))
            .collect::<Vec<_>>();
        keyed.sort();
        for (slot, (_, path)) in paths.iter_mut().zip(keyed) {
            *slot = path;
        }
    }
}

/// Whether the sources lie in more than one folder: a selection put together
/// from several shoots or several parts of one.
pub(crate) fn spans_several_folders(images: &[&ImageInfo]) -> bool {
    images
        .iter()
        .map(|info| folder_of(&info.filename))
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
    fn the_largest_capture_composes_whatever_the_order() {
        let mut images = [
            info(0, "/scans/first/DSC_0101.NEF"),
            info(1, "/scans/first/DSC_0102.NEF"),
            info(2, "/scans/second/DSC_0001.NEF"),
            info(3, "/scans/second/DSC_0002.NEF"),
            info(4, "/scans/second/DSC_0003.NEF"),
        ];
        for _ in 0..images.len() {
            images.rotate_left(1);
            let refs = images.iter().collect::<Vec<_>>();
            assert_eq!(composing_capture_ids(&refs), Some(HashSet::from([2, 3, 4])));
        }
        // Equal captures: the one whose first photo sorts first by path.
        let tied = images
            .iter()
            .filter(|info| info.id != 4)
            .collect::<Vec<_>>();
        assert_eq!(composing_capture_ids(&tied), Some(HashSet::from([0, 1])));
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
