//! Diagnostic captures that production code calls in test builds (quality_gate_runner and
//! stitch_images): native gradient ROIs and failed native boundaries. acceptance_tests
//! re-exports both, so their paths are unchanged.

use super::*;

pub(crate) fn capture_native_gradient_roi(
    world: (f64, f64),
    path: &str,
    value: f64,
    local_scale: f64,
    output: &image::Rgb32FImage,
    reference: &image::Rgb32FImage,
) {
    let Some(directory) =
        std::env::var_os("RAW_EDITOR_NATIVE_GRADIENT_DIAGNOSTICS").map(PathBuf::from)
    else {
        return;
    };
    let witness = std::env::var("RAW_EDITOR_NATIVE_GRADIENT_ROIS")
        .ok()
        .is_some_and(|text| {
            text.split(';').any(|pair| {
                let values = pair
                    .split(',')
                    .filter_map(|v| v.parse::<f64>().ok())
                    .collect::<Vec<_>>();
                values.len() == 2 && values[0] == world.0 && values[1] == world.1
            })
        });
    if !witness && value >= quality_gate::QUALITY_GRADIENT_RATIO_MIN {
        return;
    }
    fs::create_dir_all(&directory).unwrap();
    let prefix = directory.join(format!("gradient-{}-{}", world.0, world.1));
    for (name, image) in [("output", output), ("reference", reference)] {
        DynamicImage::ImageRgb32F(image.clone())
            .save(prefix.with_extension(format!("{name}.float.tiff")))
            .unwrap();
        DynamicImage::ImageRgb32F(image.clone())
            .to_rgb16()
            .save(prefix.with_extension(format!("{name}.png")))
            .unwrap();
    }
    fs::write(prefix.with_extension("json"),serde_json::to_vec_pretty(&serde_json::json!({
        "diagnostic_only":true,"world":world,"owner_path":path,"measured":value,"local_scale":local_scale,
        "threshold":quality_gate::QUALITY_GRADIENT_RATIO_MIN,
    })).unwrap()).unwrap();
}

pub(in crate::panorama_stitching) fn capture_failed_native_boundaries(
    input: &quality_gate_runner::QualityGateInput<'_>,
    report: &stack_report::QualityGateReport,
) {
    let Some(directory) =
        std::env::var_os("RAW_EDITOR_FAILED_NATIVE_BOUNDARIES").map(PathBuf::from)
    else {
        return;
    };
    fs::create_dir_all(&directory).unwrap();
    for criterion in &report.criteria {
        if criterion.name != "boundary_stroke_alignment" {
            continue;
        }
        for (index, failed) in criterion.failed.iter().enumerate() {
            let x = (failed.world.x - input.world_origin.0).round() as u32;
            let y = (failed.world.y - input.world_origin.1).round() as u32;
            let left = x.saturating_sub(48);
            let top = y.saturating_sub(48);
            let width = 96.min(input.output.width().saturating_sub(left));
            let height = 96.min(input.output.height().saturating_sub(top));
            let crop = image::imageops::crop_imm(input.output, left, top, width, height).to_image();
            let prefix = directory.join(format!("boundary-{index:03}"));
            DynamicImage::ImageRgb32F(crop.clone())
                .save(prefix.with_extension("float.tiff"))
                .unwrap();
            DynamicImage::ImageRgb32F(crop)
                .to_rgb16()
                .save(prefix.with_extension("visible.png"))
                .unwrap();
            image::imageops::crop_imm(input.coverage, left, top, width, height)
                .to_image()
                .save(prefix.with_extension("coverage.png"))
                .unwrap();
            let owners = (top..top + height)
                .flat_map(|row| {
                    (left..left + width).map(move |column| {
                        input.ownership
                            [row as usize * input.output.width() as usize + column as usize]
                    })
                })
                .collect::<Vec<_>>();
            image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(width, height, owners)
                .unwrap()
                .save(prefix.with_extension("owners.png"))
                .unwrap();
            let legend = input
                .sources
                .iter()
                .map(|source| {
                    serde_json::json!({
                        "owner":source.owner,"path":source.path,"station":source.geometry.station_id
                    })
                })
                .collect::<Vec<_>>();
            fs::write(prefix.with_extension("json"),serde_json::to_vec_pretty(&serde_json::json!({
                "diagnostic_only":true,"reported":failed,"world_origin":input.world_origin,
                "canvas":[x,y],"crop":[left,top,width,height],"crop_point":[x-left,y-top],"legend":legend,
            })).unwrap()).unwrap();
        }
    }
}
