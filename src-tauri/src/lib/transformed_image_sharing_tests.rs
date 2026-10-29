use super::{apply_all_transformations, transformed_image_into_arc};
use image::{DynamicImage, Rgb32FImage};
use std::borrow::Cow;
use std::sync::Arc;

#[test]
fn unchanged_full_resolution_images_reuse_the_decoded_arc() {
    let original = Arc::new(DynamicImage::ImageRgb32F(Rgb32FImage::new(64, 48)));
    let (unchanged, offset) =
        apply_all_transformations(Cow::Borrowed(original.as_ref()), &serde_json::json!({}));
    let shared = transformed_image_into_arc(unchanged, &original);

    assert_eq!(offset, (0.0, 0.0));
    assert!(Arc::ptr_eq(&shared, &original));
}

#[test]
fn owned_transform_results_keep_their_existing_allocation() {
    let original = Arc::new(DynamicImage::ImageRgb32F(Rgb32FImage::new(64, 48)));
    let transformed = DynamicImage::ImageRgb32F(Rgb32FImage::new(48, 64));
    let shared = transformed_image_into_arc(Cow::Owned(transformed), &original);

    assert!(!Arc::ptr_eq(&shared, &original));
    assert_eq!(shared.width(), 48);
    assert_eq!(shared.height(), 64);
}
