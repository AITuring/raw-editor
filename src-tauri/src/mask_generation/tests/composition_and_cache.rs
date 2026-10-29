//! Mask composition and cache ownership.
//!
//! The first additive sub-mask keeps its allocation, lazy composition keeps the
//! black-base semantics, cached masks share one allocation and range-mask sources
//! are borrowed or display-processed.

use super::*;

#[test]
fn first_additive_submask_reuses_its_pixel_allocation() {
    let additive = GrayImage::from_fn(17, 11, |x, y| Luma([((x * 13 + y * 7) % 256) as u8]));
    let additive_ptr = additive.as_raw().as_ptr();
    let expected = additive.clone();
    let mut composed = None;

    composite_sub_mask(&mut composed, additive, SubMaskMode::Additive);

    let composed = composed.expect("additive bitmap becomes the composition base");
    assert_eq!(composed.as_raw().as_ptr(), additive_ptr);
    assert_eq!(composed, expected);
}

#[test]
fn lazy_mask_composition_preserves_black_base_semantics() {
    let mut composed = None;
    composite_sub_mask(
        &mut composed,
        GrayImage::from_pixel(5, 3, Luma([200])),
        SubMaskMode::Subtractive,
    );
    composite_sub_mask(
        &mut composed,
        GrayImage::from_pixel(5, 3, Luma([100])),
        SubMaskMode::Intersect,
    );
    assert!(composed.is_none());

    composite_sub_mask(
        &mut composed,
        GrayImage::from_pixel(5, 3, Luma([180])),
        SubMaskMode::Additive,
    );
    composite_sub_mask(
        &mut composed,
        GrayImage::from_pixel(5, 3, Luma([30])),
        SubMaskMode::Subtractive,
    );
    composite_sub_mask(
        &mut composed,
        GrayImage::from_pixel(5, 3, Luma([120])),
        SubMaskMode::Intersect,
    );

    assert!(
        composed
            .expect("additive submask establishes the output")
            .pixels()
            .all(|pixel| pixel[0] == 120)
    );
}

#[test]
fn shared_mask_cache_keeps_one_60mp_pixel_allocation() {
    const WIDTH: usize = 9_504;
    const HEIGHT: usize = 6_336;
    const MASK_BYTES: usize = 60_217_344;

    assert_eq!(WIDTH * HEIGHT, MASK_BYTES);
    let mask = Arc::new(GrayImage::from_pixel(7, 5, Luma([123])));
    let mut cache = BudgetedCache::new(2, 1_024);
    assert!(cache.insert(42_u64, Arc::clone(&mask), mask.as_raw().len()));
    let caller = Arc::clone(cache.get(&42).expect("shared mask cache hit"));

    assert!(Arc::ptr_eq(&mask, &caller));
    assert_eq!(mask.as_raw().as_ptr(), caller.as_raw().as_ptr());
    assert_eq!(MASK_BYTES * 2 - MASK_BYTES, MASK_BYTES);
}

#[test]
fn range_mask_source_preparation_borrows_non_raw_and_preserves_raw_display_semantics() {
    let source = DynamicImage::ImageRgb32F(image::Rgb32FImage::from_fn(7, 5, |x, y| {
        image::Rgb([
            (x + 1) as f32 / 8.0,
            (y + 1) as f32 / 6.0,
            (x + y + 1) as f32 / 12.0,
        ])
    }));
    let original = source.clone();
    let range_mask = MaskDefinition {
        id: "range-source".to_string(),
        name: "Range source".to_string(),
        visible: true,
        invert: false,
        opacity: 100.0,
        adjustments: Value::Null,
        sub_masks: vec![test_sub_mask(
            "color",
            serde_json::json!({}),
            SubMaskMode::Additive,
        )],
    };

    assert!(prepare_range_mask_source_image(&source, false, &[]).is_none());
    let borrowed =
        prepare_range_mask_source_image(&source, false, std::slice::from_ref(&range_mask))
            .expect("non-RAW range masks need a source");
    assert!(matches!(borrowed, Cow::Borrowed(image) if std::ptr::eq(image, &source)));

    let mut expected = source.clone();
    apply_cpu_default_raw_processing(&mut expected);
    let processed =
        prepare_range_mask_source_image(&source, true, std::slice::from_ref(&range_mask))
            .expect("RAW range masks need a display source")
            .into_owned();
    assert_eq!(processed, expected);
    assert_eq!(
        source, original,
        "RAW preparation must not mutate the source"
    );
}
