//! Explicit, bounded test-only all-source focus comparison. Never selects owners.
use super::{QualityGateInput, SourceLoader};
use image::{DynamicImage, GrayImage, Luma, Rgb32FImage};
use nalgebra::Point2;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::PathBuf;

#[derive(Deserialize, Serialize)]
struct Window {
    x: u32,
    y: u32,
    side: u32,
}
#[derive(Deserialize)]
struct Request {
    dimensions: [u32; 2],
    origin: [f64; 2],
    windows: Vec<Window>,
}
fn validate(r: &Request, dimensions: [u32; 2], origin: [f64; 2]) -> Result<(), String> {
    if r.dimensions != dimensions
        || r.origin != origin
        || r.windows.is_empty()
        || r.windows.len() > 4
    {
        return Err("focus candidate request canvas/origin/count mismatch".into());
    }
    for w in &r.windows {
        if w.side == 0
            || w.side > 256
            || w.x.checked_add(w.side).is_none_or(|v| v > dimensions[0])
            || w.y.checked_add(w.side).is_none_or(|v| v > dimensions[1])
        {
            return Err("focus candidate window exceeds bounds".into());
        }
    }
    Ok(())
}
fn save(path: PathBuf, image: Rgb32FImage) -> Result<(), String> {
    DynamicImage::ImageRgb32F(image)
        .to_rgb16()
        .save(path)
        .map_err(|e| e.to_string())
}
pub(super) fn capture(
    input: &QualityGateInput<'_>,
    loader: &mut SourceLoader<'_>,
) -> Result<(), String> {
    let Some(request) = std::env::var_os("RAW_EDITOR_FOCUS_CANDIDATE_REQUEST") else {
        return Ok(());
    };
    let directory = PathBuf::from(
        std::env::var_os("RAW_EDITOR_FOCUS_CANDIDATE_DIR").ok_or("candidate directory missing")?,
    );
    let bytes = std::fs::read(request).map_err(|e| e.to_string())?;
    if bytes.len() > 16384 {
        return Err("candidate request too large".into());
    }
    let r: Request = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    validate(
        &r,
        [input.output.width(), input.output.height()],
        [input.world_origin.0, input.world_origin.1],
    )?;
    if input.acceptance_render_scale != 1.0 {
        return Err("candidate capture requires native scale=1".into());
    }
    std::fs::create_dir(&directory).map_err(|e| e.to_string())?;
    for (i, w) in r.windows.iter().enumerate() {
        save(
            directory.join(format!("roi-{i}-output.png")),
            image::imageops::crop_imm(input.output, w.x, w.y, w.side, w.side).to_image(),
        )?;
    }
    let mut records = Vec::new();
    let mut loads = 0;
    for source in input.sources {
        let inverse =
            (source.geometry.tile_to_world * source.geometry.member_to_anchor).try_inverse();
        let maps = r
            .windows
            .iter()
            .map(|w| {
                (0..w.side)
                    .flat_map(|y| (0..w.side).map(move |x| (x, y)))
                    .map(|(x, y)| {
                        let inverse = inverse.as_ref()?;
                        // This diagnostic deliberately requires native registration, so
                        // unsupported legacy geometry is reported instead of guessed.
                        source.native_registration.as_ref()?;
                        let world = Point2::new(
                            input.world_origin.0 + f64::from(w.x + x),
                            input.world_origin.1 + f64::from(w.y + y),
                        );
                        let p = super::source_sampling::source_point_for_reference(
                            world,
                            &source.geometry,
                            input.residual,
                            source.native_registration.as_ref(),
                            inverse,
                        )
                        .ok()?;
                        (p.x.is_finite()
                            && p.y.is_finite()
                            && p.x >= 0.0
                            && p.y >= 0.0
                            && p.x <= f64::from(source.dimensions.0.saturating_sub(1))
                            && p.y <= f64::from(source.dimensions.1.saturating_sub(1)))
                        .then_some(p)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let supported = maps.iter().any(|m| m.iter().any(Option::is_some));
        let image = if supported {
            loads += 1;
            let image = loader(source)?;
            if image.dimensions() != source.dimensions {
                return Err("candidate prepared dimensions mismatch".into());
            }
            Some(image)
        } else {
            None
        };
        for (i, (w, map)) in r.windows.iter().zip(maps).enumerate() {
            let mut reference = Rgb32FImage::new(w.side, w.side);
            let mut mask = GrayImage::new(w.side, w.side);
            let mut selected = GrayImage::new(w.side, w.side);
            let mut valid = 0;
            for (n, p) in map.iter().enumerate() {
                let x = n as u32 % w.side;
                let y = n as u32 / w.side;
                if let Some(p) = p {
                    reference.put_pixel(
                        x,
                        y,
                        crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
                            image.as_ref().unwrap(),
                            p.x,
                            p.y,
                        ),
                    );
                    mask.put_pixel(x, y, Luma([255]));
                    valid += 1;
                }
                let n = (w.y + y) as usize * input.output.width() as usize + (w.x + x) as usize;
                if input.ownership[n] == source.owner
                    && input.coverage.get_pixel(w.x + x, w.y + y).0[0] != 0
                {
                    selected.put_pixel(x, y, Luma([255]));
                }
            }
            let stem = format!("roi-{i}-owner-{}", source.owner);
            if valid > 0 {
                save(directory.join(format!("{stem}.png")), reference)?;
                mask.save(directory.join(format!("{stem}-support.png")))
                    .map_err(|e| e.to_string())?;
                selected
                    .save(directory.join(format!("{stem}-selected.png")))
                    .map_err(|e| e.to_string())?;
            }
            records.push(json!({"roi":i,"owner":source.owner,"path":source.path,"station":source.geometry.station_id,"valid_pixels":valid,"total_pixels":w.side*w.side,"stem":stem,"status":if valid==0 {"no_native_support"}else{"captured"}}));
        }
    }
    let manifest = json!({"schema":1,"status":"captured_candidate_only","final_iteration":null,"dimensions":r.dimensions,"world_origin":r.origin,"windows":r.windows,"source_loader_calls":loads,"candidates":records,"encoding":"RGB16 from display-referred prepared RGB, no extra gamma or tone; unsupported pixels zero with explicit support mask","comparison":"candidates pre-tone; output post-tone; compare focus on common support, not absolute brightness","full_quality_acceptance":false});
    std::fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .map_err(|e| e.to_string())
}
#[test]
fn bounds_reject_wrong_canvas_origin_and_overflow() {
    let mut r = Request {
        dimensions: [100, 100],
        origin: [-108.0, -49.0],
        windows: vec![Window {
            x: 0,
            y: 0,
            side: 64,
        }],
    };
    assert!(validate(&r, [100, 100], [-108.0, -49.0]).is_ok());
    assert!(validate(&r, [100, 100], [0.0, 0.0]).is_err());
    r.windows[0].x = u32::MAX;
    assert!(validate(&r, [100, 100], r.origin).is_err());
    r.windows[0].x = 0;
    r.windows[0].side = 257;
    assert!(validate(&r, [100, 100], r.origin).is_err());
}
