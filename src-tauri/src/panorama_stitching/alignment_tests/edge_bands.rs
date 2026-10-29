//! Long horizontal and vertical edge lines behind the focus warp bands: consensus across
//! distinct images and per-image deduplication of candidate lines.

use super::*;

#[test]
fn long_edge_consensus_groups_only_distinct_images() {
    let lines = vec![
        FocusHorizontalEdgeLine {
            image_id: 0,
            source_row: 0.20,
            world_x_center: 500.0,
            slope: 0.01,
            intercept: 995.0,
            median_error: 2.0,
        },
        FocusHorizontalEdgeLine {
            image_id: 1,
            source_row: 0.18,
            world_x_center: 700.0,
            slope: 0.011,
            intercept: 994.0,
            median_error: 2.0,
        },
        FocusHorizontalEdgeLine {
            image_id: 2,
            source_row: 0.22,
            world_x_center: 900.0,
            slope: 0.009,
            intercept: 996.0,
            median_error: 2.0,
        },
        FocusHorizontalEdgeLine {
            image_id: 0,
            source_row: 0.40,
            world_x_center: 500.0,
            slope: 0.01,
            intercept: 2_000.0,
            median_error: 2.0,
        },
    ];

    let clusters = focus_horizontal_edge_line_clusters(&lines, 9_504.0);
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].len(), 3);
}

#[test]
fn long_edge_duplicate_candidates_keep_the_best_fit_per_image() {
    let lines = vec![
        FocusHorizontalEdgeLine {
            image_id: 0,
            source_row: 0.20,
            world_x_center: 500.0,
            slope: 0.01,
            intercept: 995.0,
            median_error: 4.0,
        },
        FocusHorizontalEdgeLine {
            image_id: 0,
            source_row: 0.201,
            world_x_center: 500.0,
            slope: 0.0105,
            intercept: 995.2,
            median_error: 1.5,
        },
        FocusHorizontalEdgeLine {
            image_id: 0,
            source_row: 0.31,
            world_x_center: 500.0,
            slope: 0.01,
            intercept: 1_500.0,
            median_error: 2.0,
        },
    ];

    let deduplicated = deduplicate_focus_horizontal_edge_lines(lines, 9_504.0);
    assert_eq!(deduplicated.len(), 2);
    assert!(
        deduplicated
            .iter()
            .any(|line| (line.median_error - 1.5).abs() < f64::EPSILON)
    );
    assert!(
        deduplicated
            .iter()
            .any(|line| (line.source_row - 0.31).abs() < f64::EPSILON)
    );
}

#[test]
fn vertical_long_edge_consensus_groups_only_distinct_images() {
    let lines = vec![
        FocusVerticalEdgeLine {
            image_id: 0,
            source_column: 0.20,
            world_y_center: 500.0,
            slope: 0.01,
            intercept: 995.0,
            median_error: 2.0,
        },
        FocusVerticalEdgeLine {
            image_id: 1,
            source_column: 0.18,
            world_y_center: 700.0,
            slope: 0.011,
            intercept: 994.0,
            median_error: 2.0,
        },
        FocusVerticalEdgeLine {
            image_id: 2,
            source_column: 0.22,
            world_y_center: 900.0,
            slope: 0.009,
            intercept: 996.0,
            median_error: 2.0,
        },
        FocusVerticalEdgeLine {
            image_id: 0,
            source_column: 0.40,
            world_y_center: 500.0,
            slope: 0.01,
            intercept: 2_000.0,
            median_error: 2.0,
        },
    ];

    let clusters = focus_vertical_edge_line_clusters(&lines, 9_504.0);
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].len(), 3);
}

#[test]
fn vertical_edge_duplicate_candidates_keep_the_best_fit_per_image() {
    let lines = vec![
        FocusVerticalEdgeLine {
            image_id: 0,
            source_column: 0.20,
            world_y_center: 500.0,
            slope: 0.01,
            intercept: 995.0,
            median_error: 4.0,
        },
        FocusVerticalEdgeLine {
            image_id: 0,
            source_column: 0.201,
            world_y_center: 500.0,
            slope: 0.0105,
            intercept: 995.2,
            median_error: 1.5,
        },
        FocusVerticalEdgeLine {
            image_id: 0,
            source_column: 0.31,
            world_y_center: 500.0,
            slope: 0.01,
            intercept: 1_500.0,
            median_error: 2.0,
        },
    ];

    let deduplicated = deduplicate_focus_vertical_edge_lines(lines, 9_504.0);
    assert_eq!(deduplicated.len(), 2);
    assert!(
        deduplicated
            .iter()
            .any(|line| (line.median_error - 1.5).abs() < f64::EPSILON)
    );
    assert!(
        deduplicated
            .iter()
            .any(|line| (line.source_column - 0.31).abs() < f64::EPSILON)
    );
}
