use super::should_output_to_native_display;

#[test]
fn requires_an_enabled_native_display_and_supported_dimensions() {
    assert!(should_output_to_native_display(
        true, true, 3840, 2160, 16_384
    ));
    assert!(!should_output_to_native_display(
        false, true, 3840, 2160, 16_384
    ));
    assert!(!should_output_to_native_display(
        true, false, 3840, 2160, 16_384
    ));
    assert!(!should_output_to_native_display(
        true, true, 17_598, 2735, 16_384
    ));
    assert!(!should_output_to_native_display(
        true, true, 2735, 17_598, 16_384
    ));
}
