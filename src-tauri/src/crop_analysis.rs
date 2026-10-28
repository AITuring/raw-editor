#[cfg(test)]
use image::DynamicImage;

const UPRIGHT_AXIS_TOLERANCE_DEGREES: f32 = 25.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StructuralAxis {
    Horizontal,
    Vertical,
}

#[derive(Debug, Clone, Copy)]
struct StructuralLine {
    start: (f32, f32),
    end: (f32, f32),
    axis: StructuralAxis,
    support: usize,
    weight: f32,
}

fn normalize_line_angle(mut angle: f32) -> f32 {
    while angle > 90.0 {
        angle -= 180.0;
    }
    while angle <= -90.0 {
        angle += 180.0;
    }
    angle
}

mod straighten;

pub use straighten::{StraightenAnalysis, analyze_straighten};

mod structural_lines;

use structural_lines::{
    extract_structural_lines, structural_line_center, structural_line_direction,
    structural_line_length, structural_line_residual,
};

mod axis_consensus;

use axis_consensus::{
    consensus_has_independent_structure, consensus_supports_perspective,
    consensuses_are_orthogonal_peers, fit_axis_consensus,
};

mod upright;

pub use upright::{UprightAnalysis, UprightMode, analyze_upright};
#[cfg(test)]
use upright::{UprightParameters, optimize_upright, project_point};

#[cfg(test)]
mod tests;
