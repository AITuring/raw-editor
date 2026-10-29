use super::*;

mod focus_ownership;
mod native_refinement;
mod streaming_tiles;
mod tone_harmonization;

fn image_info(id: usize, image: &Rgb32FImage) -> ImageInfo {
    ImageInfo {
        id,
        filename: format!("source-{id}.png"),
        width: image.width(),
        height: image.height(),
        alignment_image: image::DynamicImage::ImageRgb32F(image.clone()).to_luma8(),
        full_image: None,
        scale_factor: 1.0,
        focal_length_35mm: None,
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
