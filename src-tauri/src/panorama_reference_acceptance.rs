//! Opt-in reference-driven focus-stack acceptance harness.
//!
//! These tests deliberately bypass feature matching.  A registration manifest
//! (usually produced from a trusted Photoshop alignment) supplies the source
//! homographies, while the production RAW loader and focus renderer still do
//! all pixel work.  The harness is ignored unless a caller supplies the
//! environment variables documented by each test.

use super::*;
use image::{DynamicImage, GenericImageView, ImageFormat, Rgb, Rgb32FImage};
use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

const REFERENCE_THUMBNAIL_LONG_SIDE: u32 = 1_800;
const REFERENCE_PREVIEW_LONG_SIDE: u32 = 2_400;

#[derive(Debug, Serialize)]
struct ExportSource {
    id: usize,
    filename: String,
    width: u32,
    height: u32,
    thumb_width: u32,
    thumb_height: u32,
    thumb_filename: String,
}

#[derive(Debug, Serialize)]
struct ExportManifest {
    sources: Vec<ExportSource>,
}

#[derive(Debug, Deserialize)]
struct ReferenceManifest {
    reference_width: u32,
    reference_height: u32,
    #[serde(default)]
    reference_preview_width: Option<u32>,
    #[serde(default)]
    reference_preview_height: Option<u32>,
    #[serde(default)]
    registration_preview_width: Option<u32>,
    #[serde(default)]
    registration_preview_height: Option<u32>,
    #[serde(default)]
    group_tone_calibration: HashMap<String, ToneCalibration>,
    sources: Vec<ReferenceSource>,
}

#[derive(Debug, Deserialize, Clone, Copy)]
struct ToneCalibration {
    gain_rgb: [f32; 3],
    offset_rgb: [f32; 3],
}

#[derive(Debug, Deserialize)]
struct ReferenceSource {
    id: usize,
    filename: String,
    width: u32,
    height: u32,
    group_id: u8,
    homography: Vec<f64>,
    #[serde(default)]
    refinement: Option<LocalRefinement>,
    #[serde(default)]
    tone_calibration: Option<ToneCalibration>,
}

#[derive(Debug, Deserialize, Clone)]
struct LocalRefinement {
    origin_x: i32,
    origin_y: i32,
    width: u32,
    height: u32,
    step: u32,
    cols: u32,
    rows: u32,
    displacements: Vec<[f32; 2]>,
}

impl LocalRefinement {
    fn delta(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let local_x = x - f64::from(self.origin_x);
        let local_y = y - f64::from(self.origin_y);
        if local_x < 0.0
            || local_y < 0.0
            || local_x > f64::from(self.width.saturating_sub(1))
            || local_y > f64::from(self.height.saturating_sub(1))
        {
            return None;
        }
        let gx = (local_x / f64::from(self.step.max(1))).clamp(0.0, self.cols as f64 - 1.0);
        let gy = (local_y / f64::from(self.step.max(1))).clamp(0.0, self.rows as f64 - 1.0);
        let x0 = gx.floor() as usize;
        let y0 = gy.floor() as usize;
        let x1 = (x0 + 1).min(self.cols.saturating_sub(1) as usize);
        let y1 = (y0 + 1).min(self.rows.saturating_sub(1) as usize);
        let fx = gx - x0 as f64;
        let fy = gy - y0 as f64;
        let at = |ix: usize, iy: usize| -> [f64; 2] {
            let value = self.displacements[iy * self.cols as usize + ix];
            [f64::from(value[0]), f64::from(value[1])]
        };
        let a = at(x0, y0);
        let b = at(x1, y0);
        let c = at(x0, y1);
        let d = at(x1, y1);
        Some((
            (a[0] * (1.0 - fx) + b[0] * fx) * (1.0 - fy) + (c[0] * (1.0 - fx) + d[0] * fx) * fy,
            (a[1] * (1.0 - fx) + b[1] * fx) * (1.0 - fy) + (c[1] * (1.0 - fx) + d[1] * fx) * fy,
        ))
    }
}

fn map_homography(matrix: &Matrix3<f64>, x: f64, y: f64) -> Option<(f64, f64)> {
    let point = matrix * nalgebra::Vector3::new(x, y, 1.0);
    (point.z.is_finite() && point.z.abs() > 1e-10).then_some((point.x / point.z, point.y / point.z))
}

fn sample_rgb32(image: &Rgb32FImage, x: f64, y: f64) -> Option<Rgb<f32>> {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x > f64::from(image.width().saturating_sub(1))
        || y > f64::from(image.height().saturating_sub(1))
    {
        return None;
    }
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(image.width().saturating_sub(1));
    let y1 = (y0 + 1).min(image.height().saturating_sub(1));
    let fx = (x - x0 as f64) as f32;
    let fy = (y - y0 as f64) as f32;
    let p00 = image.get_pixel(x0, y0).0;
    let p10 = image.get_pixel(x1, y0).0;
    let p01 = image.get_pixel(x0, y1).0;
    let p11 = image.get_pixel(x1, y1).0;
    Some(Rgb(std::array::from_fn(|channel| {
        let top = p00[channel] * (1.0 - fx) + p10[channel] * fx;
        let bottom = p01[channel] * (1.0 - fx) + p11[channel] * fx;
        top * (1.0 - fy) + bottom * fy
    })))
}

fn apply_reference_tone(image: Rgb32FImage, calibration: Option<ToneCalibration>) -> Rgb32FImage {
    let Some(calibration) = calibration else {
        return image;
    };
    let (width, height) = image.dimensions();
    Rgb32FImage::from_fn(width, height, |x, y| {
        let pixel = image.get_pixel(x, y);
        Rgb(std::array::from_fn(|channel| {
            (pixel[channel] * calibration.gain_rgb[channel] + calibration.offset_rgb[channel])
                .clamp(0.0, 1.0)
        }))
    })
}

fn prewarp_reference_source(
    source: &Rgb32FImage,
    homography: &Matrix3<f64>,
    refinement: &LocalRefinement,
    render_scale: f64,
    reference_width: u32,
    reference_height: u32,
    preview_width: u32,
    preview_height: u32,
) -> Rgb32FImage {
    let Some(inverse) = homography.try_inverse() else {
        return source.clone();
    };
    let full_to_preview_x = f64::from(preview_width) / f64::from(reference_width.max(1));
    let full_to_preview_y = f64::from(preview_height) / f64::from(reference_height.max(1));
    let preview_to_full_x = full_to_preview_x.recip();
    let preview_to_full_y = full_to_preview_y.recip();
    Rgb32FImage::from_fn(source.width(), source.height(), |x, y| {
        let Some((qx, qy)) = map_homography(homography, x as f64, y as f64) else {
            return *source.get_pixel(x, y);
        };
        let preview_x = qx / render_scale * full_to_preview_x;
        let preview_y = qy / render_scale * full_to_preview_y;
        let Some((dx, dy)) = refinement.delta(preview_x, preview_y) else {
            return *source.get_pixel(x, y);
        };
        let corrected_qx = qx + dx * preview_to_full_x * render_scale;
        let corrected_qy = qy + dy * preview_to_full_y * render_scale;
        let Some((sx, sy)) = map_homography(&inverse, corrected_qx, corrected_qy) else {
            return *source.get_pixel(x, y);
        };
        sample_rgb32(source, sx, sy).unwrap_or_else(|| *source.get_pixel(x, y))
    })
}

fn required_paths() -> Vec<String> {
    let encoded = std::env::var_os("RAW_EDITOR_FOCUS_STACK_PATHS")
        .expect("RAW_EDITOR_FOCUS_STACK_PATHS must contain platform-separated image paths");
    let paths = std::env::split_paths(&encoded)
        .map(|path| path.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    assert!(
        paths.len() >= 2,
        "at least two focus-stack paths are required"
    );
    paths
}

fn reference_workdir() -> PathBuf {
    let path = PathBuf::from(
        std::env::var_os("RAW_EDITOR_REFERENCE_WORKDIR")
            .expect("RAW_EDITOR_REFERENCE_WORKDIR must name a writable directory"),
    );
    fs::create_dir_all(&path).expect("create reference acceptance workdir");
    path
}

fn thumbnail_dimensions(width: u32, height: u32) -> (u32, u32) {
    let long_side = width.max(height).max(1) as f64;
    let scale = f64::from(REFERENCE_THUMBNAIL_LONG_SIDE) / long_side;
    (
        (f64::from(width) * scale).round().max(1.0) as u32,
        (f64::from(height) * scale).round().max(1.0) as u32,
    )
}

fn parse_render_scale() -> f64 {
    std::env::var("RAW_EDITOR_REFERENCE_RENDER_SCALE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(1.0)
        .clamp(0.01, 1.0)
}

fn load_reference_manifest() -> ReferenceManifest {
    let path = PathBuf::from(
        std::env::var_os("RAW_EDITOR_REFERENCE_MANIFEST")
            .expect("RAW_EDITOR_REFERENCE_MANIFEST must name the registration JSON"),
    );
    serde_json::from_slice(&fs::read(&path).expect("read reference registration manifest"))
        .expect("parse reference registration manifest")
}

fn save_reference_outputs(image: DynamicImage, output_path: &Path) {
    let canonical = crate::image_stack::canonicalize_image_stack_result(image);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).expect("create reference output directory");
    }
    crate::image_stack::write_srgb_tiff(&canonical, output_path)
        .expect("write 16-bit ICC reference TIFF");

    let stem = output_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("reference-stack");
    let parent = output_path.parent().unwrap_or_else(|| Path::new("."));
    let app_jpeg = parent.join(format!("{stem}.jpg"));
    crate::image_stack::write_srgb_jpeg(&canonical, &app_jpeg)
        .expect("write app-compatible reference JPEG");

    let preview = canonical.resize(
        REFERENCE_PREVIEW_LONG_SIDE,
        REFERENCE_PREVIEW_LONG_SIDE,
        image::imageops::FilterType::Lanczos3,
    );
    let preview_path = parent.join(format!("{stem}.preview2400.jpg"));
    preview
        .save_with_format(&preview_path, ImageFormat::Jpeg)
        .expect("write 2400px reference preview");
    println!(
        "reference render outputs: TIFF={}, JPEG={}, preview={}",
        output_path.display(),
        app_jpeg.display(),
        preview_path.display()
    );
}

#[test]
#[ignore = "requires RAW_EDITOR_FOCUS_STACK_PATHS and RAW_EDITOR_REFERENCE_WORKDIR"]
fn export_prepared_sources_from_env() {
    let paths = required_paths();
    let workdir = reference_workdir();
    crate::sidecar_storage::initialize(Path::new("/private/tmp/raw-editor-reference-sidecars"))
        .expect("initialize reference acceptance sidecar storage");
    let settings = AppSettings::default();
    let mut sources = Vec::with_capacity(paths.len());

    for (id, filename) in paths.iter().enumerate() {
        let prepared = load_prepared_stack_source(filename, &settings)
            .expect("load and process a reference RAW source");
        let (width, height) = prepared.image.dimensions();
        let (thumb_width, thumb_height) = thumbnail_dimensions(width, height);
        let thumb = prepared.image.resize_exact(
            thumb_width,
            thumb_height,
            image::imageops::FilterType::Lanczos3,
        );
        let source_stem = Path::new(filename)
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("source");
        let thumb_path = workdir.join(format!("{id:04}-{source_stem}.jpg"));
        thumb
            .save_with_format(&thumb_path, ImageFormat::Jpeg)
            .expect("write prepared source thumbnail");
        sources.push(ExportSource {
            id,
            filename: filename.clone(),
            width,
            height,
            thumb_width,
            thumb_height,
            thumb_filename: thumb_path.to_string_lossy().into_owned(),
        });
    }

    let manifest = workdir.join("reference-sources.json");
    fs::write(
        &manifest,
        serde_json::to_vec_pretty(&ExportManifest { sources }).expect("serialize source manifest"),
    )
    .expect("write prepared source manifest");
    println!(
        "exported {} prepared sources and thumbnails to {}",
        paths.len(),
        manifest.display()
    );
}

#[test]
#[ignore = "requires RAW_EDITOR_REFERENCE_MANIFEST and RAW_EDITOR_REFERENCE_WORKDIR"]
fn export_full_reference_sources_from_env() {
    let manifest = load_reference_manifest();
    let workdir = reference_workdir();
    crate::sidecar_storage::initialize(Path::new("/private/tmp/raw-editor-reference-sidecars"))
        .expect("initialize reference acceptance sidecar storage");
    for source in &manifest.sources {
        let output = workdir.join(format!("{:04}.tiff", source.id));
        if output.exists() {
            continue;
        }
        let prepared = load_prepared_stack_source(&source.filename, &AppSettings::default())
            .expect("develop full RAW source");
        assert_eq!(prepared.image.dimensions(), (source.width, source.height));
        let canonical = crate::image_stack::canonicalize_image_stack_result(prepared.image);
        crate::image_stack::write_srgb_tiff(&canonical, &output).expect("save developed source");
        println!("exported full source {}: {}", source.id, output.display());
    }
}

#[test]
#[ignore = "requires RAW_EDITOR_REFERENCE_MANIFEST and RAW_EDITOR_STACK_ACCEPTANCE_OUTPUT"]
fn render_reference_registered_stack_from_env() {
    let manifest = load_reference_manifest();
    assert!(
        manifest.sources.len() >= 2,
        "reference manifest needs at least two sources"
    );
    let scale = parse_render_scale();
    crate::sidecar_storage::initialize(Path::new("/private/tmp/raw-editor-reference-sidecars"))
        .expect("initialize reference acceptance sidecar storage");

    let mut images = Vec::with_capacity(manifest.sources.len());
    let mut homographies = HashMap::with_capacity(manifest.sources.len());
    let mut groups = HashMap::with_capacity(manifest.sources.len());
    let mut refinements = HashMap::with_capacity(manifest.sources.len());
    let mut tone_calibrations = HashMap::with_capacity(manifest.sources.len());
    for source in &manifest.sources {
        assert_eq!(
            source.homography.len(),
            9,
            "homography must contain nine row-major values"
        );
        let source_image = ImageInfo {
            id: source.id,
            filename: source.filename.clone(),
            width: source.width,
            height: source.height,
            alignment_image: GrayImage::new(0, 0),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm: None,
            overview_reference: false,
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: Vec::new(),
            vertical_edge_columns: Vec::new(),
        };
        images.push(source_image);
        homographies.insert(source.id, Matrix3::from_row_slice(&source.homography));
        groups.insert(source.id, source.group_id);
        if let Some(refinement) = source.refinement.clone() {
            refinements.insert(source.id, refinement);
        }
        let tone = source.tone_calibration.or_else(|| {
            manifest
                .group_tone_calibration
                .get(&source.group_id.to_string())
                .copied()
        });
        if let Some(tone) = tone {
            tone_calibrations.insert(source.id, tone);
        }
    }

    let scaled_images = images
        .iter()
        .map(|image| scaled_render_image_info(image, scale))
        .collect::<Vec<_>>();
    let scaled_homographies = scaled_source_render_homographies(&homographies, scale);
    let image_refs = scaled_images.iter().collect::<Vec<_>>();
    let preview_width = manifest
        .reference_preview_width
        .or(manifest.registration_preview_width)
        .unwrap_or(6_000);
    let preview_height = manifest
        .reference_preview_height
        .or(manifest.registration_preview_height)
        .unwrap_or(1_644);
    let app = tauri::test::mock_app();
    let mut load_image = |info: &ImageInfo| -> Result<Rgb32FImage, String> {
        let prepared = load_prepared_stack_source(&info.filename, &AppSettings::default())?;
        let dynamic = if (scale - 1.0).abs() < f64::EPSILON {
            prepared.image
        } else {
            let width = (f64::from(prepared.image.width()) * scale).round().max(1.0) as u32;
            let height = (f64::from(prepared.image.height()) * scale)
                .round()
                .max(1.0) as u32;
            prepared
                .image
                .resize_exact(width, height, image::imageops::FilterType::Lanczos3)
        };
        let mut image = apply_reference_tone(
            dynamic.to_rgb32f(),
            tone_calibrations.get(&info.id).copied(),
        );
        if let Some(refinement) = refinements.get(&info.id) {
            image = prewarp_reference_source(
                &image,
                &scaled_homographies[&info.id],
                refinement,
                scale,
                manifest.reference_width,
                manifest.reference_height,
                preview_width,
                preview_height,
            );
        }
        Ok(image)
    };
    let rendered = stitching::focus_stack_stitcher(
        &image_refs,
        &scaled_homographies,
        Projection::Planar,
        None,
        Some(&groups),
        true,
        app.handle().clone(),
        "reference-image-stack-progress",
        &mut load_image,
    )
    .expect("reference homographies should render through the production focus renderer");
    assert!(rendered.width() > 0 && rendered.height() > 0);
    let output = PathBuf::from(
        std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_OUTPUT")
            .expect("RAW_EDITOR_STACK_ACCEPTANCE_OUTPUT must name the output TIFF"),
    );
    save_reference_outputs(DynamicImage::ImageRgb32F(rendered), &output);
}
