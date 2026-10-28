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
pub(super) fn capture_test_tone_tile(tile: &tone::ToneTile) {
    let directory = TEST_TONE_CAPTURE.lock().unwrap().clone();
    let Some(directory) = directory else {
        return;
    };
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
    let record = serde_json::json!({"station":tile.station_index,"owner":tile.owner_id,
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
    for tile in tiles {
        capture_test_tone_tile(tile);
    }
    let Some(directory) = TEST_TONE_CAPTURE.lock().unwrap().clone() else {
        return;
    };
    image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(
        panorama.width(),
        panorama.height(),
        owners.to_vec(),
    )
    .unwrap()
    .save(directory.join("tone-owners.png"))
    .unwrap();
}
