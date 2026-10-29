//! The centered CFA probe and its rawler develop stages.
//!
//! Rescale, demosaic and colour-transform stages are developed on a CFA-aligned
//! crop and summarised for the report.

use rawler::imgop::develop::DemosaicAlgorithm;
use rawler::imgop::develop::Intermediate;
use rawler::imgop::develop::ProcessingStep;
use rawler::imgop::develop::RawDevelop;
use rawler::rawimage::RawImage;
use rawler::rawimage::RawImageData;
use rawler::rawimage::RawPhotometricInterpretation;
use sha2::Digest;
use sha2::Sha256;
use std::time::Instant;

use super::{Dimensions, NodeProbeReport, Point, StageSummary, TimedStageSummary, elapsed_ms};

pub(super) fn run_node_probe(id: &str, probe: &RawImage, source_origin: Point) -> NodeProbeReport {
    let (rescale, rescale_ms) = develop_stage(probe, vec![ProcessingStep::Rescale], "rescale", id);
    let (demosaic, demosaic_ms) = develop_stage(
        probe,
        vec![ProcessingStep::Rescale, ProcessingStep::Demosaic],
        "demosaic",
        id,
    );
    let (neutral_color, neutral_color_ms) = develop_stage(
        probe,
        vec![
            ProcessingStep::Rescale,
            ProcessingStep::Demosaic,
            ProcessingStep::Calibrate,
        ],
        "neutral-WB color transform",
        id,
    );
    let (as_shot_color, as_shot_color_ms) = develop_stage(
        probe,
        vec![
            ProcessingStep::Rescale,
            ProcessingStep::Demosaic,
            ProcessingStep::WhiteBalance,
            ProcessingStep::Calibrate,
        ],
        "as-shot-WB color transform",
        id,
    );

    assert_eq!(
        stage_channels(&rescale),
        1,
        "{id} rescale must preserve the mosaic"
    );
    for (stage, name) in [
        (&demosaic, "demosaic"),
        (&neutral_color, "neutral color transform"),
        (&as_shot_color, "as-shot color transform"),
    ] {
        assert_eq!(
            stage_channels(stage),
            3,
            "{id} {name} must produce three channels"
        );
    }

    let color_transform_delta = mean_absolute_delta(&demosaic, &neutral_color);
    let white_balance_delta = mean_absolute_delta(&neutral_color, &as_shot_color);
    assert!(
        color_transform_delta > 1.0e-5,
        "{id} camera-to-RGB color transform had no measurable effect"
    );
    assert!(
        white_balance_delta > 1.0e-5,
        "{id} as-shot white balance had no measurable effect"
    );

    NodeProbeReport {
        source_origin,
        source_dimensions: Dimensions::from_usize(probe.width, probe.height),
        rescale: TimedStageSummary {
            elapsed_ms: rescale_ms,
            output: summarize_stage(&rescale, &format!("{id} rescale")),
        },
        demosaic_camera_rgb: TimedStageSummary {
            elapsed_ms: demosaic_ms,
            output: summarize_stage(&demosaic, &format!("{id} demosaic")),
        },
        neutral_white_balance_color_transform: TimedStageSummary {
            elapsed_ms: neutral_color_ms,
            output: summarize_stage(&neutral_color, &format!("{id} neutral color")),
        },
        as_shot_white_balance_color_transform: TimedStageSummary {
            elapsed_ms: as_shot_color_ms,
            output: summarize_stage(&as_shot_color, &format!("{id} as-shot color")),
        },
        color_transform_mean_absolute_delta: color_transform_delta,
        white_balance_mean_absolute_delta: white_balance_delta,
    }
}

pub(super) fn develop_stage(
    raw: &RawImage,
    steps: Vec<ProcessingStep>,
    stage_name: &str,
    id: &str,
) -> (Intermediate, f64) {
    let start = Instant::now();
    let output = RawDevelop {
        steps,
        demosaic_algorithm: DemosaicAlgorithm::Quality,
    }
    .develop_intermediate(raw)
    .unwrap_or_else(|error| panic!("{id} {stage_name} failed: {error}"));
    (output, elapsed_ms(start))
}

pub(super) fn centered_cfa_probe(raw: &RawImage, requested_edge: usize) -> (RawImage, Point) {
    let (period_x, period_y) = match &raw.photometric {
        RawPhotometricInterpretation::Cfa(config) => {
            (config.cfa.width.max(1), config.cfa.height.max(1))
        }
        _ => (1, 1),
    };
    let probe_width = aligned_length(requested_edge.min(raw.width), period_x);
    let probe_height = aligned_length(requested_edge.min(raw.height), period_y);
    let origin_x = ((raw.width - probe_width) / 2 / period_x) * period_x;
    let origin_y = ((raw.height - probe_height) / 2 / period_y) * period_y;
    let row_values = probe_width * raw.cpp;

    let data = match &raw.data {
        RawImageData::Integer(values) => {
            let mut cropped = Vec::with_capacity(probe_width * probe_height * raw.cpp);
            for y in origin_y..origin_y + probe_height {
                let start = (y * raw.width + origin_x) * raw.cpp;
                cropped.extend_from_slice(&values[start..start + row_values]);
            }
            RawImageData::Integer(cropped)
        }
        RawImageData::Float(values) => {
            let mut cropped = Vec::with_capacity(probe_width * probe_height * raw.cpp);
            for y in origin_y..origin_y + probe_height {
                let start = (y * raw.width + origin_x) * raw.cpp;
                cropped.extend_from_slice(&values[start..start + row_values]);
            }
            RawImageData::Float(cropped)
        }
    };

    let probe = RawImage {
        camera: raw.camera.clone(),
        make: raw.make.clone(),
        model: raw.model.clone(),
        clean_make: raw.clean_make.clone(),
        clean_model: raw.clean_model.clone(),
        width: probe_width,
        height: probe_height,
        cpp: raw.cpp,
        bps: raw.bps,
        wb_coeffs: raw.wb_coeffs,
        whitelevel: raw.whitelevel.clone(),
        blacklevel: raw.blacklevel.clone(),
        xyz_to_cam: raw.xyz_to_cam,
        photometric: raw.photometric.clone(),
        active_area: None,
        crop_area: None,
        blackareas: Vec::new(),
        orientation: raw.orientation,
        data,
        color_matrix: raw.color_matrix.clone(),
        dng_tags: raw.dng_tags.clone(),
    };

    (
        probe,
        Point {
            x: origin_x as u32,
            y: origin_y as u32,
        },
    )
}

pub(super) fn aligned_length(length: usize, period: usize) -> usize {
    let aligned = length - length % period;
    assert!(aligned > 0, "CFA-aligned probe must not be empty");
    aligned
}

pub(super) fn summarize_stage(stage: &Intermediate, label: &str) -> StageSummary {
    let dimensions = stage.dim();
    let channels = stage_channels(stage);
    let values = stage_values(stage);
    let mut hasher = Sha256::new();
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut channel_sums = vec![0.0f64; channels];
    for (index, value) in values.iter().enumerate() {
        assert!(value.is_finite(), "{label} contains a non-finite value");
        hasher.update(value.to_bits().to_le_bytes());
        min = min.min(*value);
        max = max.max(*value);
        channel_sums[index % channels] += *value as f64;
    }
    let pixels = values.len() / channels;
    StageSummary {
        dimensions: Dimensions::from_usize(dimensions.w, dimensions.h),
        channels,
        sha256: hex::encode(hasher.finalize()),
        min,
        max,
        channel_means: channel_sums
            .into_iter()
            .map(|sum| sum / pixels as f64)
            .collect(),
    }
}

pub(super) fn stage_values(stage: &Intermediate) -> Vec<f32> {
    match stage {
        Intermediate::Monochrome(pixels) => pixels.data.clone(),
        Intermediate::ThreeColor(pixels) => pixels
            .data
            .iter()
            .flat_map(|pixel| pixel.iter().copied())
            .collect(),
        Intermediate::FourColor(pixels) => pixels
            .data
            .iter()
            .flat_map(|pixel| pixel.iter().copied())
            .collect(),
    }
}

pub(super) fn stage_channels(stage: &Intermediate) -> usize {
    match stage {
        Intermediate::Monochrome(_) => 1,
        Intermediate::ThreeColor(_) => 3,
        Intermediate::FourColor(_) => 4,
    }
}

pub(super) fn mean_absolute_delta(left: &Intermediate, right: &Intermediate) -> f64 {
    assert_eq!(left.dim(), right.dim(), "stage dimensions differ");
    assert_eq!(
        stage_channels(left),
        stage_channels(right),
        "stage channel counts differ"
    );
    let left = stage_values(left);
    let right = stage_values(right);
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / left.len() as f64
}
