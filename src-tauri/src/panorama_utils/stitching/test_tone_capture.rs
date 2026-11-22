//! Test-only capture of tone tiles, contour probes and seam grids.
//!
//! Writes diagnostic images when a test sets a capture directory.

#[cfg(test)]
use super::super::stack_pipeline::tone;
use image::GrayImage;
use image::Rgb;
use image::Rgb32FImage;
use rayon::prelude::*;

#[cfg(test)]
pub(super) static TEST_TONE_CAPTURE: std::sync::Mutex<Option<std::path::PathBuf>> =
    std::sync::Mutex::new(None);

#[cfg(test)]
pub(crate) fn set_test_tone_capture_directory(path: Option<std::path::PathBuf>) {
    *TEST_TONE_CAPTURE.lock().unwrap() = path;
}

#[cfg(test)]
fn station_filter_from_env(value: Option<std::ffi::OsString>) -> Option<Vec<usize>> {
    let value = value?;
    let text = value.to_string_lossy();
    let mut stations = text
        .split([',', ' ', '\t', '\n'])
        .filter(|part| !part.is_empty())
        .filter_map(|part| part.parse::<usize>().ok())
        .collect::<Vec<_>>();
    stations.sort_unstable();
    stations.dedup();
    (!stations.is_empty()).then_some(stations)
}

#[cfg(test)]
fn selected_station(station: usize, filter: Option<&[usize]>) -> bool {
    filter.is_none_or(|stations| stations.binary_search(&station).is_ok())
}

#[cfg(test)]
fn station_label(station: usize, filter: Option<&[usize]>) -> Option<usize> {
    let labels = station_filter_from_env(std::env::var_os("RAW_EDITOR_TEST_TONE_STATION_LABELS"))?;
    let index = filter?.binary_search(&station).ok()?;
    labels.get(index).copied()
}

#[cfg(test)]
pub(super) fn capture_test_tone_tile(tile: &tone::ToneTile) {
    let directory = TEST_TONE_CAPTURE.lock().unwrap().clone();
    let Some(directory) = directory else {
        return;
    };
    let station_filter = station_filter_from_env(std::env::var_os("RAW_EDITOR_TEST_TONE_STATIONS"));
    if !selected_station(tile.station_index, station_filter.as_deref()) {
        return;
    }
    let directory = directory.as_path();
    std::fs::create_dir_all(directory).unwrap();
    if let Some(native) = tile.native_detail.as_ref() {
        use std::io::Write;
        let mut file = std::io::BufWriter::new(
            std::fs::File::create(
                directory.join(format!("tone-owner-{:03}.native.f32", tile.owner_id)),
            )
            .unwrap(),
        );
        for value in native.as_raw() {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        file.flush().unwrap();
    }
    let record = serde_json::json!({"station":tile.station_index,"station_label":station_label(tile.station_index, station_filter.as_deref()),"owner":tile.owner_id,
                "world_origin":tile.world_origin,"world_size":tile.world_size,"world_stride":tile.world_stride,
                "width":tile.low.width(),"height":tile.low.height(),"low":tile.low.as_raw(),
                "validity":tile.validity.as_raw(),"cell_mean":tile.cell_mean.as_raw(),"cell_coverage":tile.cell_coverage});
    std::fs::write(
        directory.join(format!("tone-owner-{:03}.json", tile.owner_id)),
        serde_json::to_vec(&record).unwrap(),
    )
    .unwrap();
}

#[cfg(test)]
pub(super) fn capture_test_contour_probe(
    station: usize,
    track: usize,
    points: &[[f64; 2]],
    sample: impl Fn(f64, f64) -> Option<Rgb<f32>> + Sync,
) {
    let Some(directory) = TEST_TONE_CAPTURE.lock().unwrap().clone() else {
        return;
    };
    if points.is_empty() {
        return;
    }
    let left = (points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min) - 64.0).floor() as i32;
    let right = (points
        .iter()
        .map(|p| p[1])
        .fold(f64::NEG_INFINITY, f64::max)
        + 64.0)
        .ceil() as i32;
    let top = points[0][0].floor() as i32;
    let bottom = points.last().unwrap()[0].ceil() as i32;
    let (width, height) = (
        (right - left + 1).max(1) as u32,
        (bottom - top + 1).max(1) as u32,
    );
    if u64::from(width) * u64::from(height) > 4_000_000 {
        return;
    }
    let mut rgb = Rgb32FImage::new(width, height);
    let mut mask = GrayImage::new(width, height);
    rgb.as_mut()
        .par_chunks_mut(width as usize * 3)
        .zip(mask.as_mut().par_chunks_mut(width as usize))
        .enumerate()
        .for_each(|(y, (row, covered))| {
            for x in 0..width as usize {
                if let Some(p) = sample(left as f64 + x as f64, top as f64 + y as f64) {
                    row[x * 3..x * 3 + 3].copy_from_slice(&p.0);
                    covered[x] = 255;
                }
            }
        });
    std::fs::create_dir_all(&directory).unwrap();
    let prefix = directory.join(format!("native-contour-{station:03}-{track:02}"));
    image::DynamicImage::ImageRgb32F(rgb)
        .save(prefix.with_extension("float.tiff"))
        .unwrap();
    mask.save(prefix.with_extension("coverage.png")).unwrap();
    std::fs::write(
        prefix.with_extension("json"),
        serde_json::to_vec(&serde_json::json!({
        "station":station,"world_origin":[left,top],"points":points,"diagnostic_only":true}))
        .unwrap(),
    )
    .unwrap();
}

#[cfg(test)]
pub(super) fn capture_test_seam_grid(owner: u16, record: serde_json::Value) {
    let Some(directory) = TEST_TONE_CAPTURE.lock().unwrap().clone() else {
        return;
    };
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join(format!("seam-owner-{owner:03}.json")),
        serde_json::to_vec(&record).unwrap(),
    )
    .unwrap();
}

#[cfg(test)]
pub(super) fn capture_test_tone_fields(
    panorama: &Rgb32FImage,
    owners: &[u16],
    tiles: &[tone::ToneTile],
) {
    let station_filter = station_filter_from_env(std::env::var_os("RAW_EDITOR_TEST_TONE_STATIONS"));
    for tile in tiles {
        if selected_station(tile.station_index, station_filter.as_deref()) {
            capture_test_tone_tile(tile);
        }
    }
    let Some(directory) = TEST_TONE_CAPTURE.lock().unwrap().clone() else {
        return;
    };
    std::fs::create_dir_all(&directory).unwrap();
    let selected_owner_ids = tiles
        .iter()
        .filter(|tile| selected_station(tile.station_index, station_filter.as_deref()))
        .map(|tile| tile.owner_id)
        .collect::<std::collections::BTreeSet<_>>();
    let filtered_owners = owners
        .iter()
        .map(|owner| {
            selected_owner_ids
                .contains(owner)
                .then_some(*owner)
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    let owner_plane = if station_filter.is_some() {
        "tone-owners.filtered.png"
    } else {
        "tone-owners.png"
    };
    image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(
        panorama.width(),
        panorama.height(),
        filtered_owners,
    )
    .unwrap()
    .save(directory.join(owner_plane))
    .unwrap();
    if let Some(stations) = station_filter {
        let mut selected = tiles
            .iter()
            .filter(|tile| stations.binary_search(&tile.station_index).is_ok())
            .map(|tile| tile.station_index)
            .collect::<Vec<_>>();
        selected.sort_unstable();
        selected.dedup();
        let field_files = tiles
            .iter()
            .filter(|tile| stations.binary_search(&tile.station_index).is_ok())
            .map(|tile| format!("tone-owner-{:03}.json", tile.owner_id))
            .collect::<Vec<_>>();
        let missing = stations
            .iter()
            .copied()
            .filter(|station| selected.binary_search(station).is_err())
            .collect::<Vec<_>>();
        let manifest = serde_json::json!({
            "schema": 1,
            "diagnostic_only": true,
            "station_filter": stations,
            "station_labels": stations.iter().map(|station| serde_json::json!({
                "local": station,
                "label": station_label(*station, Some(&stations)),
            })).collect::<Vec<_>>(),
            "selected_stations": selected,
            "missing_stations": missing,
            "owner_ids": selected_owner_ids,
            "field_files": field_files,
            "owner_plane": owner_plane,
            "dimensions": [panorama.width(), panorama.height()],
        });
        std::fs::write(
            directory.join("tone-fields-filter-manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn station_filter_parses_and_deduplicates() {
        assert_eq!(
            super::station_filter_from_env(Some("39,34 39\t35".into())),
            Some(vec![34, 35, 39])
        );
        assert_eq!(super::station_filter_from_env(Some("bad".into())), None);
        assert!(super::selected_station(35, Some(&[34, 35, 39])));
        assert!(!super::selected_station(36, Some(&[34, 35, 39])));
    }
}
