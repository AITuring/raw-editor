use super::*;

mod focus_fusion;
mod geometry_sampling;
mod panorama_blending;
mod station_seams;

fn geometry_test_image(id: usize, focal_length_35mm: Option<f64>) -> ImageInfo {
    ImageInfo {
        id,
        filename: format!("geometry-{id}.jpg"),
        width: 1_000,
        height: 800,
        alignment_image: GrayImage::new(1, 1),
        full_image: None,
        scale_factor: 1.0,
        focal_length_35mm,
        source_footprint: None,
        overview_reference: false,
        features: Vec::new(),
        top_features: Vec::new(),
        foreground_range: None,
        foreground_mask: None,
        horizontal_edge_rows: Vec::new(),
        vertical_edge_columns: Vec::new(),
    }
}
