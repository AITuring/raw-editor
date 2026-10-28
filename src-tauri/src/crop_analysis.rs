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
mod tests {
    use super::*;
    use image::{GrayImage, Luma, Rgb, RgbImage};
    use imageproc::drawing::draw_line_segment_mut;

    #[test]
    fn detects_tilted_horizontal_structure() {
        let mut image = GrayImage::from_pixel(640, 420, Luma([12]));
        let slope = 4.0f32.to_radians().tan();

        for base_y in [90.0, 205.0, 320.0] {
            draw_line_segment_mut(
                &mut image,
                (28.0, base_y),
                (612.0, base_y + 584.0 * slope),
                Luma([245]),
            );
        }

        let result = analyze_straighten(&DynamicImage::ImageLuma8(image));
        assert!(
            result.detected,
            "analysis should detect the synthetic horizon: {result:?}"
        );
        assert!(
            (result.angle + 4.0).abs() <= 0.8,
            "unexpected correction: {result:?}"
        );
    }

    #[test]
    fn detects_tilted_vertical_structure() {
        let mut image = GrayImage::from_pixel(640, 420, Luma([12]));
        let lean = 3.0f32.to_radians().tan();

        for base_x in [135.0, 320.0, 505.0] {
            draw_line_segment_mut(
                &mut image,
                (base_x, 25.0),
                (base_x + 370.0 * lean, 395.0),
                Luma([245]),
            );
        }

        let result = analyze_straighten(&DynamicImage::ImageLuma8(image));
        assert!(
            result.detected,
            "analysis should detect the synthetic verticals: {result:?}"
        );
        assert!(
            (result.angle - 3.0).abs() <= 0.8,
            "unexpected correction: {result:?}"
        );
    }

    #[test]
    fn ignores_images_without_structural_lines() {
        let image = DynamicImage::ImageLuma8(GrayImage::from_pixel(320, 240, Luma([128])));
        let result = analyze_straighten(&image);
        assert!(!result.detected);
        assert_eq!(result.angle, 0.0);
    }

    fn synthetic_grid_lines(
        width: f32,
        height: f32,
        distortion: UprightParameters,
    ) -> Vec<StructuralLine> {
        let mut lines = Vec::new();
        for x in [width * 0.2, width * 0.4, width * 0.6, width * 0.8] {
            lines.push(StructuralLine {
                start: project_point((x, height * 0.08), width, height, distortion).unwrap(),
                end: project_point((x, height * 0.92), width, height, distortion).unwrap(),
                axis: StructuralAxis::Vertical,
                support: 240,
                weight: 240.0,
            });
        }
        for y in [height * 0.2, height * 0.4, height * 0.6, height * 0.8] {
            lines.push(StructuralLine {
                start: project_point((width * 0.08, y), width, height, distortion).unwrap(),
                end: project_point((width * 0.92, y), width, height, distortion).unwrap(),
                axis: StructuralAxis::Horizontal,
                support: 240,
                weight: 240.0,
            });
        }
        lines
    }

    #[test]
    fn upright_optimizer_reverses_vertical_and_horizontal_keystone() {
        let width = 800.0;
        let height = 600.0;
        let distortion = UprightParameters {
            vertical: 28.0,
            horizontal: -22.0,
            ..UprightParameters::default()
        };
        let lines = synthetic_grid_lines(width, height, distortion);
        let result = optimize_upright(&lines, width, height, true, true, 0.0);

        assert!(
            (result.vertical + distortion.vertical).abs() <= 2.0,
            "{result:?}"
        );
        assert!(
            (result.horizontal + distortion.horizontal).abs() <= 2.0,
            "{result:?}"
        );
        assert!(result.rotation.abs() <= 0.5, "{result:?}");
    }

    #[test]
    fn upright_optimizer_levels_rotated_structure() {
        let width = 800.0;
        let height = 600.0;
        let distortion = UprightParameters {
            rotation: 6.0,
            ..UprightParameters::default()
        };
        let lines = synthetic_grid_lines(width, height, distortion);
        let result = optimize_upright(&lines, width, height, false, false, 0.0);

        assert!(
            (result.rotation + distortion.rotation).abs() <= 0.3,
            "{result:?}"
        );
        assert_eq!(result.vertical, 0.0);
        assert_eq!(result.horizontal, 0.0);
    }

    #[test]
    fn upright_analysis_detects_projected_grid() {
        let width = 800u32;
        let height = 600u32;
        let distortion = UprightParameters {
            vertical: 24.0,
            horizontal: -16.0,
            ..UprightParameters::default()
        };
        let mut image = GrayImage::from_pixel(width, height, Luma([14]));
        for line in synthetic_grid_lines(width as f32, height as f32, distortion) {
            draw_line_segment_mut(&mut image, line.start, line.end, Luma([242]));
        }

        let result = analyze_upright(&DynamicImage::ImageLuma8(image), UprightMode::Full, 0);
        assert!(result.detected, "{result:?}");
        assert!(
            (result.vertical + distortion.vertical).abs() <= 7.0,
            "{result:?}"
        );
        assert!(
            (result.horizontal + distortion.horizontal).abs() <= 7.0,
            "{result:?}"
        );
    }

    #[test]
    fn upright_modes_respect_their_axis_contract() {
        let width = 800u32;
        let height = 600u32;
        let distortion = UprightParameters {
            rotation: 5.0,
            vertical: 24.0,
            horizontal: -16.0,
        };
        let mut image = GrayImage::from_pixel(width, height, Luma([14]));
        for line in synthetic_grid_lines(width as f32, height as f32, distortion) {
            draw_line_segment_mut(&mut image, line.start, line.end, Luma([242]));
        }
        let image = DynamicImage::ImageLuma8(image);

        let auto = analyze_upright(&image, UprightMode::Auto, 0);
        let level = analyze_upright(&image, UprightMode::Level, 0);
        let vertical = analyze_upright(&image, UprightMode::Vertical, 0);
        let full = analyze_upright(&image, UprightMode::Full, 0);

        assert!(auto.detected, "{auto:?}");
        assert!(
            (auto.rotation + distortion.rotation).abs() <= 2.0,
            "{auto:?}"
        );
        assert!(
            (auto.vertical + distortion.vertical).abs() <= 6.0,
            "{auto:?}"
        );
        assert!(
            (auto.horizontal + distortion.horizontal).abs() <= 6.0,
            "{auto:?}"
        );
        assert!(level.detected, "{level:?}");
        assert_eq!(level.vertical, 0.0);
        assert_eq!(level.horizontal, 0.0);
        assert!(
            (level.rotation + distortion.rotation).abs() <= 4.0,
            "{level:?}"
        );
        assert!(vertical.detected, "{vertical:?}");
        assert_eq!(vertical.horizontal, 0.0);
        assert!(
            (vertical.rotation + distortion.rotation).abs() <= 2.0,
            "{vertical:?}"
        );
        assert!(
            (vertical.vertical + distortion.vertical).abs() <= 6.0,
            "{vertical:?}"
        );
        assert!(full.detected, "{full:?}");
        assert!(
            (full.rotation + distortion.rotation).abs() <= 2.0,
            "{full:?}"
        );
        assert!(
            (full.vertical + distortion.vertical).abs() <= 6.0,
            "{full:?}"
        );
        assert!(
            (full.horizontal + distortion.horizontal).abs() <= 6.0,
            "{full:?}"
        );
    }

    #[test]
    fn upright_uses_distributed_coloured_rectangles_as_geometry_evidence() {
        let width = 800u32;
        let height = 600u32;
        let distortion = UprightParameters {
            rotation: 4.0,
            ..UprightParameters::default()
        };
        // The two colours deliberately have similar luma. Grayscale-only edge
        // detection largely loses these marks, while channel edges retain the
        // repeated rectangular geometry used by stamps, windows and tiles.
        let mut image = RgbImage::from_pixel(width, height, Rgb([166, 105, 70]));
        for (center_x, center_y) in [
            (96.0f32, 92.0f32),
            (304.0, 128.0),
            (520.0, 84.0),
            (704.0, 142.0),
            (142.0, 258.0),
            (366.0, 224.0),
            (646.0, 284.0),
            (82.0, 424.0),
            (282.0, 372.0),
            (488.0, 448.0),
            (714.0, 394.0),
            (190.0, 536.0),
            (410.0, 514.0),
            (650.0, 548.0),
        ] {
            let corners = [
                (center_x - 29.0, center_y - 29.0),
                (center_x + 29.0, center_y - 29.0),
                (center_x + 29.0, center_y + 29.0),
                (center_x - 29.0, center_y + 29.0),
            ]
            .map(|point| project_point(point, width as f32, height as f32, distortion).unwrap());
            for index in 0..corners.len() {
                draw_line_segment_mut(
                    &mut image,
                    corners[index],
                    corners[(index + 1) % corners.len()],
                    Rgb([214, 65, 65]),
                );
            }
        }
        let image = DynamicImage::ImageRgb8(image);

        let auto = analyze_upright(&image, UprightMode::Auto, 0);
        assert!(auto.detected, "{auto:?}");
        assert!(
            (auto.rotation + distortion.rotation).abs() <= 2.0,
            "{auto:?}"
        );
        assert_eq!(auto.vertical, 0.0);
        assert_eq!(auto.horizontal, 0.0);

        let level = analyze_upright(&image, UprightMode::Level, 0);
        assert!(level.detected, "{level:?}");
        assert!(
            (level.rotation + distortion.rotation).abs() <= 2.0,
            "{level:?}"
        );
        assert_eq!(level.vertical, 0.0);
        assert_eq!(level.horizontal, 0.0);

        let vertical = analyze_upright(&image, UprightMode::Vertical, 0);
        assert!(vertical.detected, "{vertical:?}");
        assert!(
            (vertical.rotation + distortion.rotation).abs() <= 2.0,
            "{vertical:?}"
        );
        assert_eq!(vertical.vertical, 0.0);
        assert_eq!(vertical.horizontal, 0.0);
    }

    #[test]
    fn upright_rejects_disconnected_decorative_marks() {
        let width = 600u32;
        let height = 1000u32;
        let mut image = GrayImage::from_pixel(width, height, Luma([18]));
        for (base_x, slope) in [(130.0f32, -0.12f32), (300.0, 0.0), (470.0, 0.12)] {
            for y in (40..940).step_by(65) {
                let y = y as f32;
                let x = base_x + (y - height as f32 * 0.5) * slope;
                draw_line_segment_mut(
                    &mut image,
                    (x, y),
                    (x + slope * 22.0, y + 22.0),
                    Luma([240]),
                );
            }
        }

        let image = DynamicImage::ImageLuma8(image);
        for mode in [
            UprightMode::Auto,
            UprightMode::Level,
            UprightMode::Vertical,
            UprightMode::Full,
        ] {
            let result = analyze_upright(&image, mode, 0);
            assert!(
                !result.detected,
                "decorative marks are not {mode:?} perspective guides: {result:?}"
            );
            assert_eq!(result.rotation, 0.0);
            assert_eq!(result.vertical, 0.0);
            assert_eq!(result.horizontal, 0.0);
        }
    }

    #[test]
    fn vertical_mode_does_not_use_horizontal_lines_as_vertical_evidence() {
        let mut image = GrayImage::from_pixel(640, 420, Luma([12]));
        let slope = 5.0f32.to_radians().tan();
        for base_y in [90.0, 205.0, 320.0] {
            draw_line_segment_mut(
                &mut image,
                (28.0, base_y),
                (612.0, base_y + 584.0 * slope),
                Luma([245]),
            );
        }
        let image = DynamicImage::ImageLuma8(image);

        let vertical = analyze_upright(&image, UprightMode::Vertical, 0);
        assert!(!vertical.detected, "{vertical:?}");
        assert_eq!(vertical.rotation, 0.0);
        assert_eq!(vertical.vertical, 0.0);
        assert_eq!(vertical.horizontal, 0.0);

        let level = analyze_upright(&image, UprightMode::Level, 0);
        assert!(level.detected, "{level:?}");
        assert!((level.rotation + 5.0).abs() <= 0.8, "{level:?}");
    }
}
