//! Opt-in test diagnostics. Export the actual sampling geometry and ownership
//! decisions so a failed real-image ROI can be replayed without aligning and
//! rendering the entire stack again. Never compiled into the application.
use super::*;
use serde_json::json;
use std::path::PathBuf;

fn directory() -> Option<PathBuf> {
    std::env::var_os("RAW_EDITOR_MOSAIC_DIAGNOSTICS").map(PathBuf::from)
}

pub(super) fn capture_layer(
    index: usize,
    sampler: &LayerSampler<'_>,
    tone: &Field<3>,
    decision: &GrayImage,
    aw: u32,
    ah: u32,
) {
    let Some(dir) = directory() else { return };
    std::fs::create_dir_all(&dir).expect("create mosaic diagnostics directory");
    let projection = match sampler.projection {
        Projection::Planar => "planar",
        Projection::Cylindrical => "cylindrical",
        Projection::Spherical => "spherical",
    };
    let metadata = json!({
        "version": 1, "index": index,
        "filename": sampler.info.filename,
        "source_width": sampler.info.width, "source_height": sampler.info.height,
        "source_divisor": sampler.source_divisor,
        "focal_length_35mm": sampler.info.focal_length_35mm,
        "projection": projection,
        "inverse_column_major": sampler.inverse.as_slice(),
        "offset": [sampler.offset.0, sampler.offset.1],
        "left": sampler.left, "top": sampler.top, "scale": sampler.scale,
        "analysis_width": aw, "analysis_height": ah,
        "residual": {"width": sampler.residual.width, "height": sampler.residual.height,
            "step": sampler.residual.step, "values": sampler.residual.values},
        "tone": {"width": tone.width, "height": tone.height,
            "step": tone.step, "values": tone.values},
    });
    std::fs::write(
        dir.join(format!("layer-{index:03}.json")),
        serde_json::to_vec(&metadata).expect("serialize layer diagnostics"),
    )
    .expect("write layer diagnostics");
    decision
        .save(dir.join(format!("layer-{index:03}.ownership.png")))
        .expect("write ownership diagnostics");
}

pub(super) fn capture_crop(mask: &GrayImage) {
    let Some(dir) = directory() else { return };
    // Same largest-covered-rectangle traversal as the output crop, including
    // its strict area tie rule. Coordinates are before the final crop.
    let (width, height) = mask.dimensions();
    let mut heights = vec![0usize; width as usize];
    let mut stack = Vec::<usize>::new();
    let mut best = [0usize, 0, width as usize, height as usize];
    let mut area = 0;
    for y in 0..height as usize {
        for (x, h) in heights.iter_mut().enumerate() {
            *h = if mask.get_pixel(x as u32, y as u32)[0] > 0 {
                *h + 1
            } else {
                0
            };
        }
        stack.clear();
        for x in 0..=width as usize {
            let current = if x < width as usize { heights[x] } else { 0 };
            while let Some(&bar) = stack.last() {
                if heights[bar] <= current {
                    break;
                }
                stack.pop();
                let left = stack.last().map_or(0, |&i| i + 1);
                let candidate = (x - left) * heights[bar];
                if candidate > area {
                    area = candidate;
                    best = [left, y + 1 - heights[bar], x - left, heights[bar]];
                }
            }
            stack.push(x);
        }
    }
    std::fs::write(
        dir.join("canvas.json"),
        serde_json::to_vec(&json!({
            "canvas_width": width, "canvas_height": height,
            "crop": {"left": best[0], "top": best[1], "width": best[2], "height": best[3]}
        }))
        .unwrap(),
    )
    .expect("write crop diagnostics");
}

fn restore_field<const N: usize>(value: &serde_json::Value) -> Field<N> {
    Field {
        width: value["width"].as_u64().unwrap() as usize,
        height: value["height"].as_u64().unwrap() as usize,
        step: value["step"].as_f64().unwrap(),
        values: value["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| std::array::from_fn(|i| row[i].as_f64().unwrap()))
            .collect(),
    }
}

#[test]
#[ignore = "requires a captured real mosaic and a native canvas ROI"]
fn replay_mosaic_roi_from_env() {
    crate::sidecar_storage::initialize(std::path::Path::new(
        "/private/tmp/raw-editor-stack-sidecars",
    ))
    .expect("initialize diagnostic sidecar storage");
    let dir =
        PathBuf::from(std::env::var_os("RAW_EDITOR_MOSAIC_REPLAY").expect("capture directory"));
    let roi = std::env::var("RAW_EDITOR_MOSAIC_ROI")
        .expect("x,y,width,height in uncropped canvas pixels")
        .split(',')
        .map(|v| v.parse::<u32>().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(roi.len(), 4);
    let (rx, ry, width, height) = (roi[0], roi[1], roi[2], roi[3]);
    assert!(width > 0 && height > 0 && u64::from(width) * u64::from(height) <= 16_000_000);
    let output = std::env::var_os("RAW_EDITOR_MOSAIC_REPLAY_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| dir.join(format!("roi-{rx}-{ry}")));
    std::fs::create_dir_all(&output).unwrap();
    let mut base = Rgb32FImage::new(width, height);
    let mut covered = GrayImage::new(width, height);
    for index in 0..10_000 {
        let path = dir.join(format!("layer-{index:03}.json"));
        if !path.exists() {
            break;
        }
        let j: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let left = j["left"].as_u64().unwrap() as u32;
        let top = j["top"].as_u64().unwrap() as u32;
        let scale = j["scale"].as_f64().unwrap();
        let aw = j["analysis_width"].as_u64().unwrap() as u32;
        let ah = j["analysis_height"].as_u64().unwrap() as u32;
        if rx + width <= left
            || ry + height <= top
            || rx as f64 >= left as f64 + aw as f64 * scale
            || ry as f64 >= top as f64 + ah as f64 * scale
        {
            continue;
        }
        let filename = j["filename"].as_str().unwrap().to_string();
        let bytes = std::fs::read(&filename).unwrap();
        let mut source = crate::image_loader::load_base_image_from_bytes(
            &bytes,
            &filename,
            false,
            &crate::app_settings::AppSettings::default(),
            None,
        )
        .unwrap()
        .to_rgb32f();
        let divisor = j["source_divisor"].as_f64().unwrap();
        let mut current_divisor = 1.0;
        while current_divisor < divisor {
            source = downsample_rgb_half(&source);
            current_divisor *= 2.0;
        }
        let info = ImageInfo {
            id: index,
            filename: filename.clone(),
            width: j["source_width"].as_u64().unwrap() as u32,
            height: j["source_height"].as_u64().unwrap() as u32,
            alignment_image: GrayImage::new(1, 1),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm: j["focal_length_35mm"].as_f64(),
            features: vec![],
            top_features: vec![],
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: vec![],
            vertical_edge_columns: vec![],
        };
        assert_eq!(
            j["projection"].as_str().unwrap(),
            "planar",
            "ROI replay currently supports planar mosaics"
        );
        let inverse = j["inverse_column_major"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>();
        let sampler = LayerSampler {
            info: &info,
            source: &source,
            source_divisor: divisor,
            inverse: Matrix3::from_column_slice(&inverse),
            projection: Projection::Planar,
            offset: (
                j["offset"][0].as_f64().unwrap(),
                j["offset"][1].as_f64().unwrap(),
            ),
            left,
            top,
            scale,
            residual: restore_field(&j["residual"]),
        };
        let tone: Field<3> = restore_field(&j["tone"]);
        let decision = image::open(dir.join(format!("layer-{index:03}.ownership.png")))
            .unwrap()
            .into_luma8();
        let mut candidate = Rgb32FImage::new(width, height);
        let mut selection = GrayImage::new(width, height);
        candidate
            .as_mut()
            .par_chunks_mut(width as usize * 3)
            .zip(selection.as_mut().par_chunks_mut(width as usize))
            .enumerate()
            .for_each(|(y, (row, selected))| {
                for x in 0..width {
                    let lx = rx as f64 + x as f64 - left as f64;
                    let ly = ry as f64 + y as f64 - top as f64;
                    if lx < 0.0 || ly < 0.0 || lx >= aw as f64 * scale || ly >= ah as f64 * scale {
                        continue;
                    }
                    let Some(pixel) = sampler.sample(lx, ly) else {
                        continue;
                    };
                    let ax = lx / scale;
                    let ay = ly / scale;
                    let dx = ((ax / aw as f64 * decision.width() as f64) as u32)
                        .min(decision.width() - 1);
                    let dy = ((ay / ah as f64 * decision.height() as f64) as u32)
                        .min(decision.height() - 1);
                    row[x as usize * 3..x as usize * 3 + 3]
                        .copy_from_slice(&adjusted(pixel, tone.at(ax, ay)).0);
                    selected[x as usize] = if covered.get_pixel(x, y as u32)[0] == 0
                        || decision.get_pixel(dx, dy)[0] > 0
                    {
                        255
                    } else {
                        128
                    };
                }
            });
        let mut csv =
            String::from("canvas_x,canvas_y,base_focus,candidate_focus,raw_preference,chosen\n");
        let size = (aw.max(ah) as f64 / SELECTION_LONG_SIDE as f64).max(1.0);
        let native = size * scale;
        for dy in 0..decision.height() {
            for dx in 0..decision.width() {
                let lx = (dx as f64 + 0.5) * native;
                let ly = (dy as f64 + 0.5) * native;
                let gx = left as f64 + lx;
                let gy = top as f64 + ly;
                if gx < rx as f64 + 32.0
                    || gy < ry as f64 + 32.0
                    || gx >= (rx + width) as f64 - 32.0
                    || gy >= (ry + height) as f64 - 32.0
                {
                    continue;
                }
                let bf = cell_focus(
                    |x, y| rgb_at(&base, x - rx as f64, y - ry as f64),
                    gx,
                    gy,
                    native,
                );
                let cf = cell_focus(
                    |x, y| {
                        sampler
                            .sample(x, y)
                            .map(|p| adjusted(p, tone.at(lx / scale, ly / scale)))
                    },
                    lx,
                    ly,
                    native,
                );
                let preference = ((cf + 0.002) / (bf + 0.002)).ln().clamp(-2.0, 2.0) - 0.06;
                csv.push_str(&format!(
                    "{gx},{gy},{bf},{cf},{preference},{}\n",
                    decision.get_pixel(dx, dy)[0]
                ));
            }
        }
        std::fs::write(output.join(format!("layer-{index:03}.focus.csv")), csv).unwrap();
        if std::env::var_os("RAW_EDITOR_MOSAIC_RESELECT").is_some()
            && selection.pixels().all(|p| p[0] != 0)
            && covered.pixels().all(|p| p[0] != 0)
        {
            // Replay focus selection at the original native grid spacing.
            // Registration/tone stay frozen at their recorded values. The
            // ROI boundary is a guard, so inspect the interior, then verify
            // any promising change with the normal full-stack pipeline.
            let aligned_info = ImageInfo {
                id: index,
                filename: filename.clone(),
                width,
                height,
                alignment_image: GrayImage::new(1, 1),
                full_image: None,
                scale_factor: 1.0,
                focal_length_35mm: None,
                features: vec![],
                top_features: vec![],
                foreground_range: None,
                foreground_mask: None,
                horizontal_edge_rows: vec![],
                vertical_edge_columns: vec![],
            };
            let aligned_sampler = LayerSampler {
                info: &aligned_info,
                source: &candidate,
                source_divisor: 1.0,
                inverse: Matrix3::identity(),
                projection: Projection::Planar,
                offset: (0.0, 0.0),
                left: 0,
                top: 0,
                scale: 1.0,
                residual: Field::new(width, height, 48.0),
            };
            let selected = ownership_grid(
                &base,
                &covered,
                &aligned_sampler,
                &Field::new(width, height, 56.0),
                width,
                height,
                native,
            );
            for y in 0..height {
                for x in 0..width {
                    let dx = ((x as f64 / width as f64 * selected.width() as f64) as u32)
                        .min(selected.width() - 1);
                    let dy = ((y as f64 / height as f64 * selected.height() as f64) as u32)
                        .min(selected.height() - 1);
                    selection.put_pixel(
                        x,
                        y,
                        Luma([if selected.get_pixel(dx, dy)[0] > 0 {
                            255
                        } else {
                            128
                        }]),
                    );
                }
            }
            println!("ROI layer {index}: recomputed ownership at {native:.2}px cell spacing");
        }
        let mut accepted = 0;
        for y in 0..height {
            for x in 0..width {
                if selection.get_pixel(x, y)[0] == 255 {
                    base.put_pixel(x, y, *candidate.get_pixel(x, y));
                    covered.put_pixel(x, y, Luma([255]));
                    accepted += 1;
                }
            }
        }
        let save = |name: &str, im: &Rgb32FImage| {
            let image = crate::image_stack::canonicalize_image_stack_result(
                image::DynamicImage::ImageRgb32F(im.clone()),
            );
            image
                .save(output.join(format!("layer-{index:03}.{name}.png")))
                .unwrap();
        };
        save("candidate", &candidate);
        save("after", &base);
        selection
            .save(output.join(format!("layer-{index:03}.selection.png")))
            .unwrap();
        println!(
            "ROI layer {index}: {filename}; accepted {accepted}/{}",
            width * height
        );
    }
    crate::image_stack::canonicalize_image_stack_result(image::DynamicImage::ImageRgb32F(base))
        .save(output.join("replayed.png"))
        .unwrap();
}
