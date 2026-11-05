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
            source_footprint: None,
            overview_reference: false,
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
            native_mapping: None,
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
                source_footprint: None,
                overview_reference: false,
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
                native_mapping: None,
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
                false,
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

/// Append the cells of the last streaming ownership decision that fall inside
/// a canvas ROI to a CSV (`RAW_EDITOR_OWNERSHIP_TRACE=<csv>;x0,y0,x1,y1`, in
/// canvas pixels before the final crop, at the render scale), so a soft or
/// dark region can be traced to the evidence every layer saw there.
#[allow(clippy::too_many_arguments)]
pub(super) fn write_ownership_trace(
    layer: usize,
    filename: &str,
    left: u32,
    top: u32,
    layer_scale: f64,
    base_owner: &GrayImage,
    owner_names: &HashMap<u8, String>,
) {
    let Some(trace) = super::source_ownership::trace::take() else {
        return;
    };
    let Ok(spec) = std::env::var("RAW_EDITOR_OWNERSHIP_TRACE") else {
        return;
    };
    // `<csv>;x0,y0,x1,y1[;stride]`: a stride keeps one cell in `stride` along
    // either grid axis, so a whole canvas can be traced.
    let mut parts = spec.split(';');
    let (Some(path), Some(roi)) = (parts.next(), parts.next()) else {
        return;
    };
    let stride = parts
        .next()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .unwrap_or(1)
        .max(1);
    let roi = roi
        .split(',')
        .filter_map(|value| value.trim().parse::<f64>().ok())
        .collect::<Vec<_>>();
    if roi.len() != 4 || base_owner.width() == 0 || base_owner.height() == 0 {
        return;
    }
    let short = |name: &str| {
        let file = name.rsplit('/').next().unwrap_or(name);
        file.split('.').next().unwrap_or(file).to_string()
    };
    let mut rows = String::new();
    for (index, &label) in trace.labels.iter().enumerate() {
        let grid_x = index as u32 % trace.grid_width;
        let grid_y = index as u32 / trace.grid_width;
        if !grid_x.is_multiple_of(stride) || !grid_y.is_multiple_of(stride) {
            continue;
        }
        let analysis_x = (f64::from(grid_x) + 0.5) * trace.size;
        let analysis_y = (f64::from(grid_y) + 0.5) * trace.size;
        let canvas_x = f64::from(left) + analysis_x * layer_scale;
        let canvas_y = f64::from(top) + analysis_y * layer_scale;
        if canvas_x < roi[0] || canvas_y < roi[1] || canvas_x >= roi[2] || canvas_y >= roi[3] {
            continue;
        }
        let owner = base_owner.get_pixel(
            (analysis_x as u32).min(base_owner.width() - 1),
            (analysis_y as u32).min(base_owner.height() - 1),
        )[0];
        let owner_name = owner_names
            .get(&owner)
            .map(|name| short(name))
            .unwrap_or_else(|| owner.to_string());
        let (base_focus, candidate_focus) = trace
            .focus
            .get(index)
            .copied()
            .unwrap_or((f64::NAN, f64::NAN));
        rows.push_str(&format!(
            "{layer},{},{grid_x},{grid_y},{canvas_x:.0},{canvas_y:.0},{owner_name},{},{base_focus:.5},{candidate_focus:.5},{:.3},{:.3},{:.3},{:.3},{label}\n",
            short(filename),
            trace.fixed[index],
            trace.agreement.get(index).copied().unwrap_or(f64::NAN),
            trace.focus_term.get(index).copied().unwrap_or(f64::NAN),
            trace.preference[index],
            trace.disagreement[index],
        ));
    }
    if rows.is_empty() {
        return;
    }
    use std::io::Write;
    let fresh = !std::path::Path::new(path).exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open ownership trace");
    if fresh {
        file.write_all(b"layer,file,gx,gy,canvas_x,canvas_y,base_owner,fixed,base_focus,candidate_focus,agreement,focus_term,preference,disagreement,label\n")
            .expect("write ownership trace header");
    }
    file.write_all(rows.as_bytes())
        .expect("write ownership trace");
}

/// Write what one streaming layer's ownership decision saw inside canvas ROIs
/// (`RAW_EDITOR_LAYER_PROBE=<dir>;x0,y0,x1,y1[;x0,y0,x1,y1...]`, canvas pixels
/// before the final crop): the canvas before the layer, the tone-adjusted
/// candidate and the cells it took, as 8-bit PNGs at canvas resolution.
#[allow(clippy::too_many_arguments)]
pub(super) fn write_layer_probe(
    layer: usize,
    filename: &str,
    store: &StreamingMosaicStore,
    sampler: &LayerSampler<'_>,
    tone: &Field<3>,
    region: (u32, u32, u32, u32),
    decision: &GrayImage,
    analysis: (u32, u32),
) -> Result<(), String> {
    let Ok(spec) = std::env::var("RAW_EDITOR_LAYER_PROBE") else {
        return Ok(());
    };
    let mut parts = spec.split(';');
    let Some(directory) = parts.next() else {
        return Ok(());
    };
    let directory = PathBuf::from(directory);
    let (left, right, top, bottom) = region;
    let stem = std::path::Path::new(filename)
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    let to_u8 = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    for (roi_index, roi) in parts.enumerate() {
        let roi = roi
            .split(',')
            .filter_map(|value| value.trim().parse::<u32>().ok())
            .collect::<Vec<_>>();
        if roi.len() != 4 {
            continue;
        }
        let (x0, y0) = (roi[0].max(left), roi[1].max(top));
        let (x1, y1) = (roi[2].min(right + 1), roi[3].min(bottom + 1));
        if x0 >= x1 || y0 >= y1 {
            continue;
        }
        std::fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        let (width, height) = (x1 - x0, y1 - y0);
        let mut base = image::RgbImage::new(width, height);
        for tile_row in y0 / store.tile_size..=(y1 - 1) / store.tile_size {
            for tile_column in x0 / store.tile_size..=(x1 - 1) / store.tile_size {
                let (tile_left, tile_top, tile_width, tile_height) =
                    store.tile_extent(tile_column, tile_row);
                let (tile, mask) = store.load_tile(tile_column, tile_row)?;
                for y in y0.max(tile_top)..y1.min(tile_top + tile_height) {
                    for x in x0.max(tile_left)..x1.min(tile_left + tile_width) {
                        let (tile_x, tile_y) = (x - tile_left, y - tile_top);
                        if mask.get_pixel(tile_x, tile_y)[0] == 0 {
                            continue;
                        }
                        let pixel = tile.get_pixel(tile_x, tile_y);
                        base.put_pixel(
                            x - x0,
                            y - y0,
                            image::Rgb([to_u8(pixel[0]), to_u8(pixel[1]), to_u8(pixel[2])]),
                        );
                    }
                }
            }
        }
        let mut candidate = image::RgbImage::new(width, height);
        let mut take = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let local_x = f64::from(x0 + x - left);
                let local_y = f64::from(y0 + y - top);
                let (analysis_x, analysis_y) = (local_x / sampler.scale, local_y / sampler.scale);
                if let Some(pixel) = sampler.sample(local_x, local_y) {
                    let pixel = adjusted(pixel, tone.at(analysis_x, analysis_y));
                    candidate.put_pixel(
                        x,
                        y,
                        image::Rgb([to_u8(pixel[0]), to_u8(pixel[1]), to_u8(pixel[2])]),
                    );
                }
                let cell_x = ((analysis_x / f64::from(analysis.0.max(1)))
                    * f64::from(decision.width())) as u32;
                let cell_y = ((analysis_y / f64::from(analysis.1.max(1)))
                    * f64::from(decision.height())) as u32;
                let value = decision
                    .get_pixel(
                        cell_x.min(decision.width() - 1),
                        cell_y.min(decision.height() - 1),
                    )
                    .0[0];
                take.put_pixel(x, y, Luma([value]));
            }
        }
        let prefix = directory.join(format!("roi{roi_index}-L{layer:03}-{stem}"));
        for (suffix, result) in [
            ("base", base.save(prefix.with_extension("base.png"))),
            ("cand", candidate.save(prefix.with_extension("cand.png"))),
            ("take", take.save(prefix.with_extension("take.png"))),
        ] {
            result.map_err(|error| format!("write layer probe {suffix}: {error}"))?;
        }
    }
    Ok(())
}

/// Compare the composed output's detail with the sharpest frame on the
/// soft-frame detail grid (`RAW_EDITOR_DETAIL_DEFICIT=<prefix>`): writes
/// `<prefix>.csv` with every measured cell and `<prefix>.png`, one pixel per
/// cell, grey where the output keeps at least 85 % of the sharpest kept frame,
/// yellow from 70 %, red below, blue where no kept frame measured the cell.
/// A red cell is blur that a better owner choice could still remove.
pub(super) fn write_detail_deficit(
    output: &Rgb32FImage,
    crop: (u32, u32, u32, u32),
    grid: &super::soft_frames::DetailGrid,
    details: &[super::soft_frames::FrameDetail],
    images: &[&ImageInfo],
    skipped: &std::collections::HashSet<usize>,
    prefix: &std::path::Path,
) -> Result<(), String> {
    let measured = super::soft_frames::measure_image_detail(output, (crop.0, crop.1), grid);
    let count = grid.cell_count();
    let mut best_kept = vec![(0.0f64, usize::MAX); count];
    let mut best_all = vec![(0.0f64, usize::MAX); count];
    for (frame, detail) in details.iter().enumerate() {
        let kept = !skipped.contains(&images[frame].id);
        for &(cell, value) in &detail.cells {
            if value > best_all[cell].0 {
                best_all[cell] = (value, frame);
            }
            if kept && value > best_kept[cell].0 {
                best_kept[cell] = (value, frame);
            }
        }
    }
    let name = |frame: usize| {
        images.get(frame).map_or_else(String::new, |info| {
            std::path::Path::new(&info.filename)
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_default()
        })
    };
    let (columns, rows) = grid.cell_dimensions();
    let mut map = image::RgbImage::from_pixel(columns as u32, rows as u32, image::Rgb([0, 0, 0]));
    let mut csv = String::from(
        "cell,canvas_x,canvas_y,output,best_kept,best_kept_frame,best_all,best_all_frame,ratio\n",
    );
    let mut rows_out = Vec::new();
    for &(cell, value) in &measured {
        let (centre_x, centre_y) = grid.cell_centre(cell);
        let (kept, kept_frame) = best_kept[cell];
        let (all, all_frame) = best_all[cell];
        let ratio = if kept > 0.0 { value / kept } else { f64::NAN };
        let colour = if !ratio.is_finite() {
            [40, 80, 255]
        } else if ratio >= 0.85 {
            [110, 110, 110]
        } else if ratio >= 0.7 {
            [255, 210, 40]
        } else {
            [255, 40, 40]
        };
        map.put_pixel(
            (cell % columns) as u32,
            (cell / columns) as u32,
            image::Rgb(colour),
        );
        csv.push_str(&format!(
            "{cell},{centre_x:.0},{centre_y:.0},{value:.5},{kept:.5},{},{all:.5},{},{ratio:.3}\n",
            name(kept_frame),
            name(all_frame)
        ));
        rows_out.push((ratio, cell, centre_x, centre_y, value, kept, kept_frame));
    }
    std::fs::write(prefix.with_extension("csv"), csv)
        .map_err(|error| format!("write detail deficit table: {error}"))?;
    map.save(prefix.with_extension("png"))
        .map_err(|error| format!("write detail deficit map: {error}"))?;
    let finite = rows_out
        .iter()
        .filter(|row| row.0.is_finite())
        .collect::<Vec<_>>();
    let below = |limit: f64| finite.iter().filter(|row| row.0 < limit).count();
    println!(
        "  - Detail deficit: {} cells measured, output below 85% / 70% of the sharpest kept frame in {} / {}",
        finite.len(),
        below(0.85),
        below(0.70)
    );
    let mut worst = finite;
    worst.sort_by(|a, b| a.0.total_cmp(&b.0));
    for row in worst.iter().take(12) {
        println!(
            "    - cell {} at ({:.0}, {:.0}): output {:.3}, {} {:.3} (ratio {:.2})",
            row.1,
            row.2,
            row.3,
            row.4,
            name(row.6),
            row.5,
            row.0
        );
    }
    Ok(())
}

/// Write the final streaming owner plane as a colour-coded PNG (one colour per
/// owner label, quarter resolution) with a JSON sidecar naming each label's
/// source, so a region on the delivered canvas can be traced to its frame.
pub(super) fn write_streaming_owner_map(
    store: &StreamingMosaicStore,
    crop: (u32, u32, u32, u32),
    owner_names: &HashMap<u8, String>,
    path: &std::path::Path,
) -> Result<(), String> {
    const STEP: u32 = 4;
    let (crop_left, crop_top, crop_width, crop_height) = crop;
    let width = crop_width.div_ceil(STEP);
    let height = crop_height.div_ceil(STEP);
    let mut labels = vec![0u8; width as usize * height as usize];
    for tile_row in 0..store.tile_rows {
        for tile_column in 0..store.tile_columns {
            let (tile_left, tile_top, tile_width, tile_height) =
                store.tile_extent(tile_column, tile_row);
            let owner = store.load_owner_tile(tile_column, tile_row);
            for y in (0..tile_height).filter(|y| (tile_top + y).wrapping_sub(crop_top) % STEP == 0)
            {
                let canvas_y = tile_top + y;
                if canvas_y < crop_top || canvas_y >= crop_top + crop_height {
                    continue;
                }
                for x in 0..tile_width {
                    let canvas_x = tile_left + x;
                    if canvas_x < crop_left
                        || canvas_x >= crop_left + crop_width
                        || (canvas_x - crop_left) % STEP != 0
                    {
                        continue;
                    }
                    let index = ((canvas_y - crop_top) / STEP * width
                        + (canvas_x - crop_left) / STEP) as usize;
                    labels[index] = owner.get_pixel(x, y)[0];
                }
            }
        }
    }
    let colour = |label: u8| {
        if label == 0 {
            return [0u8, 0, 0];
        }
        let hash = (u32::from(label)).wrapping_mul(2_654_435_761);
        [
            64 + (hash >> 24) as u8 % 192,
            64 + (hash >> 16) as u8 % 192,
            64 + (hash >> 8) as u8 % 192,
        ]
    };
    let image = image::RgbImage::from_fn(width, height, |x, y| {
        image::Rgb(colour(labels[(y * width + x) as usize]))
    });
    image
        .save(path)
        .map_err(|error| format!("write owner map {}: {error}", path.display()))?;
    let names = owner_names
        .iter()
        .map(|(label, name)| {
            (
                label.to_string(),
                json!({"file": name, "rgb": colour(*label)}),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    std::fs::write(
        path.with_extension("json"),
        serde_json::to_vec_pretty(&json!({"step": STEP, "crop": [crop_left, crop_top, crop_width, crop_height], "owners": names}))
            .map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("write owner map names: {error}"))?;
    println!("  - Owner map written to {}", path.display());
    Ok(())
}

/// The detail repair cell by cell (`RAW_EDITOR_REPAIR_TRACE=<prefix>`):
/// `<prefix>.csv` with every cell the pass looked at, and `<prefix>.png`, one
/// pixel per repair-grid cell: grey where the canvas already held its
/// sharpest source, green where a repair brought it within 80 %, yellow
/// where a repair helped but fell short, red where the donor could not be
/// aligned, magenta where it disagreed at low frequency, blue where it added
/// too little detail.
pub(super) fn write_repair_trace(
    grid: &super::soft_frames::DetailGrid,
    outcome: &super::RepairOutcome,
    images: &[&ImageInfo],
    prefix: &std::path::Path,
) -> Result<(), String> {
    use super::CellOutcome;
    let name = |frame: usize| {
        images.get(frame).map_or_else(String::new, |info| {
            std::path::Path::new(&info.filename)
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_default()
        })
    };
    let (columns, rows) = grid.cell_dimensions();
    let mut map = image::RgbImage::from_pixel(columns as u32, rows as u32, image::Rgb([0, 0, 0]));
    for (cell, value) in outcome.detail_after.iter().enumerate() {
        if value.is_finite() {
            map.put_pixel(
                (cell % columns) as u32,
                (cell / columns) as u32,
                image::Rgb([90, 90, 90]),
            );
        }
    }
    let mut csv = String::from("cell,canvas_x,canvas_y,best,donor,before,after,outcome,strain\n");
    for cell in &outcome.cells {
        let (centre_x, centre_y) = grid.cell_centre(cell.cell);
        let colour = match cell.outcome {
            CellOutcome::Repaired if cell.after >= super::REPAIR_DEFICIT_RATIO * cell.best => {
                [40, 220, 60]
            }
            CellOutcome::Replaced if cell.after >= super::REPAIR_DEFICIT_RATIO * cell.best => {
                [40, 220, 220]
            }
            CellOutcome::Repaired | CellOutcome::Replaced => [255, 210, 40],
            CellOutcome::Unaligned => [255, 40, 40],
            CellOutcome::Disagrees => [230, 40, 230],
            CellOutcome::NoGain => [60, 110, 255],
        };
        map.put_pixel(
            (cell.cell % columns) as u32,
            (cell.cell / columns) as u32,
            image::Rgb(colour),
        );
        csv.push_str(&format!(
            "{},{centre_x:.0},{centre_y:.0},{:.5},{},{:.5},{:.5},{:?},{:.4}\n",
            cell.cell,
            cell.best,
            name(cell.donor),
            cell.before,
            cell.after,
            cell.outcome,
            cell.strain
        ));
    }
    std::fs::write(prefix.with_extension("csv"), csv)
        .map_err(|error| format!("write repair trace: {error}"))?;
    map.save(prefix.with_extension("png"))
        .map_err(|error| format!("write repair map: {error}"))?;
    // Every measured cell, `<prefix>.grid.csv`, to tune the soft-region
    // report offline.
    let mut before = outcome.detail_after.clone();
    for cell in &outcome.cells {
        before[cell.cell] = cell.before;
    }
    let mut grid_csv =
        String::from("cell,canvas_x,canvas_y,before,after,best,best_frame,level,clipped\n");
    for (cell, after) in outcome.detail_after.iter().enumerate() {
        if !after.is_finite() {
            continue;
        }
        let (centre_x, centre_y) = grid.cell_centre(cell);
        let (best, frame) = outcome.best[cell];
        grid_csv.push_str(&format!(
            "{cell},{centre_x:.0},{centre_y:.0},{:.5},{after:.5},{best:.5},{},{:.4},{:.4}\n",
            before[cell],
            name(frame),
            outcome.level[cell],
            outcome.clipped[cell],
        ));
    }
    std::fs::write(prefix.with_extension("grid.csv"), grid_csv)
        .map_err(|error| format!("write repair grid: {error}"))?;
    Ok(())
}

/// Images of one repair window (`RAW_EDITOR_REPAIR_PROBE=<dir>;x0,y0,x1,y1;...`,
/// canvas rectangles): the canvas, the donor at its global pose and the donor
/// after dense alignment, as 8-bit luma, plus the displacement at the window
/// centre in the file name.
pub(super) fn write_repair_probe(
    window: (usize, usize, usize, usize),
    info: &ImageInfo,
    base: &super::Plane,
    moving: &super::Plane,
    margin: usize,
    aligned: &super::Plane,
    displacement: &super::Displacement,
) {
    let Ok(spec) = std::env::var("RAW_EDITOR_REPAIR_PROBE") else {
        return;
    };
    let mut parts = spec.split(';');
    let Some(directory) = parts.next() else {
        return;
    };
    let (x0, y0, x1, y1) = window;
    let hit = parts.any(|rect| {
        let values = rect
            .split(',')
            .filter_map(|value| value.trim().parse::<usize>().ok())
            .collect::<Vec<_>>();
        values.len() == 4 && values[0] < x1 && values[2] > x0 && values[1] < y1 && values[3] > y0
    });
    if !hit {
        return;
    }
    let _ = std::fs::create_dir_all(directory);
    let to_image = |plane: &super::Plane, offset: usize| {
        let (width, height) = (plane.width - 2 * offset, plane.height - 2 * offset);
        image::GrayImage::from_fn(width as u32, height as u32, |x, y| {
            let value = plane.values[(y as usize + offset) * plane.width + x as usize + offset];
            image::Luma([if value.is_finite() {
                (value.clamp(0.0, 1.0) * 255.0) as u8
            } else {
                0
            }])
        })
    };
    let stem = std::path::Path::new(&info.filename)
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    let [dx, dy] = displacement.at((x1 - x0) as f64 * 0.5, (y1 - y0) as f64 * 0.5);
    let name = format!(
        "{directory}/w{x0}_{y0}_{x1}_{y1}-{stem}-d{dx:+.0}_{dy:+.0}-m{:.2}",
        displacement.matched_share
    );
    let _ = to_image(base, 0).save(format!("{name}.canvas.png"));
    let _ = to_image(moving, margin).save(format!("{name}.donor.png"));
    let _ = to_image(aligned, 0).save(format!("{name}.aligned.png"));
}
