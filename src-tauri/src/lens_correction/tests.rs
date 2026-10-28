use super::*;

fn stack_metadata() -> std::collections::HashMap<String, String> {
    [
        ("Make", "NIKON CORPORATION"),
        ("Model", "NIKON Z 7_2"),
        ("LensModel", "NIKKOR Z 24-120mm f/4 S"),
        ("FocalLength", "24"),
        ("FocalLengthIn35mmFilm", "24"),
    ]
    .into_iter()
    .map(|(k, v)| (k.into(), v.into()))
    .collect()
}

#[test]
fn calibrated_footprint_excludes_padding_and_retains_real_black_pixels() {
    let calibration = stack_lens_calibration(&stack_metadata(), (8256, 5504)).unwrap();
    let white = image::Rgb32FImage::from_pixel(600, 400, image::Rgb([1.0; 3]));
    let corrected = calibration.correct(&white).unwrap();
    let footprint = StackLensFootprint {
        calibration,
        original: white.dimensions(),
        corrected: corrected.dimensions(),
    };
    let mut supported = 0;
    let mut padding = 0;
    for y in 0..corrected.height() {
        for x in 0..corrected.width() {
            if footprint.contains(x as f64, y as f64, corrected.dimensions()) {
                supported += 1;
                assert!(corrected.get_pixel(x, y)[0] > 0.99);
            } else if corrected.get_pixel(x, y)[0] == 0.0 {
                padding += 1;
            }
        }
    }
    assert!(supported > 600 * 350 && padding > 1000);
    let black = image::Rgb32FImage::new(600, 400);
    let corrected_black = calibration.correct(&black).unwrap();
    assert_eq!(
        corrected_black.get_pixel(corrected.width() / 2, corrected.height() / 2)[0],
        0.0
    );
    assert!(footprint.contains(
        (corrected.width() / 2) as f64,
        (corrected.height() / 2) as f64,
        corrected.dimensions()
    ));
    assert!(footprint.contains(150.0, 100.0, (300, 200)));
}

#[test]
fn stack_calibration_requires_exact_known_hardware_and_sensor_format() {
    let mut metadata = stack_metadata();
    let calibration = stack_lens_calibration(&metadata, (8256, 5504)).unwrap();
    assert!((calibration.a - 0.0428727).abs() < 1e-7);
    assert!((calibration.b + 0.140952).abs() < 1e-7);
    assert!((calibration.c - 0.0855397).abs() < 1e-7);
    metadata.insert("LensModel".into(), "NIKKOR Z 24-70mm f/4 S unknown".into());
    assert!(stack_lens_calibration(&metadata, (8256, 5504)).is_none());
    metadata = stack_metadata();
    metadata.insert("Model".into(), "NIKON Z 50".into());
    assert!(stack_lens_calibration(&metadata, (8256, 5504)).is_none());
    metadata = stack_metadata();
    assert!(stack_lens_calibration(&metadata, (8000, 5504)).is_none());
    metadata.insert("FocalLengthIn35mmFilm".into(), "36".into());
    assert!(stack_lens_calibration(&metadata, (8256, 5504)).is_none());
    metadata.remove("FocalLengthIn35mmFilm");
    assert!(stack_lens_calibration(&metadata, (8256, 5504)).is_none());
    metadata.insert("FocalLength".into(), "200".into());
    assert!(stack_lens_calibration(&metadata, (8256, 5504)).is_none());
}

#[test]
fn stack_vignetting_corrects_linear_sensor_values_before_gamma() {
    let mut metadata = stack_metadata();
    metadata.insert("FNumber".into(), "f/4".into());
    let calibration = stack_lens_calibration(&metadata, (8256, 5504)).unwrap();
    let k = calibration.vignetting.unwrap();
    assert!((k[0] + 0.5030).abs() < 1e-6);
    assert!((k[1] + 0.2477).abs() < 1e-6);
    assert!((k[2] + 0.0487).abs() < 1e-6);
    let mut image = image::Rgb32FImage::from_pixel(600, 400, image::Rgb([0.1, 0.2, 0.3]));
    calibration.correct_linear_illumination(&mut image);
    let center = image.get_pixel(300, 200);
    assert!((center[0] - 0.1).abs() < 1e-5);
    let corner = image.get_pixel(0, 0);
    assert!(corner[0] > 0.48 && corner[0] < 0.51);
    assert!(
        corner[2] > 1.0,
        "linear headroom must survive until the display curve"
    );
    assert!((corner[1] / corner[0] - 2.0).abs() < 1e-5);
    metadata.insert("FNumber".into(), "f/5".into());
    let interpolated = stack_lens_calibration(&metadata, (8256, 5504))
        .unwrap()
        .vignetting
        .unwrap();
    assert!(
        interpolated
            .iter()
            .all(|coefficient| coefficient.is_finite())
    );
    let falloff = 1.0 + interpolated.iter().sum::<f64>();
    assert!(falloff > 0.05 && falloff < 0.5);
}

#[test]
fn stack_ptlens_radius_uses_original_short_side_and_full_strength() {
    let calibration = stack_lens_calibration(&stack_metadata(), (8256, 5504)).unwrap();
    assert!((calibration.distorted_radius(1.0) - 1.0).abs() < 1e-12);
    let distorted = calibration.distorted_radius(1.5);
    assert!((distorted - 1.45260376875).abs() < 1e-6);
    assert!((calibration.undistorted_radius(distorted).unwrap() - 1.5).abs() < 1e-9);
}

#[test]
fn stack_lens_canvas_preserves_original_boundary_and_masks_empty_corners() {
    let calibration = stack_lens_calibration(&stack_metadata(), (8256, 5504)).unwrap();
    let source = image::Rgb32FImage::from_pixel(600, 400, image::Rgb([0.4, 0.5, 0.6]));
    let corrected = calibration.correct(&source).unwrap();
    assert!(corrected.width() > source.width());
    assert!(corrected.height() > source.height());
    assert_eq!(corrected.get_pixel(0, 0).0, [0.0; 3]);
    let center = corrected.get_pixel(corrected.width() / 2, corrected.height() / 2);
    assert!((center[0] - 0.4).abs() < 1e-6);
    let unit = 200.0;
    for (x, y) in [(0.0, 0.0), (599.0, 399.0), (299.5, 0.0), (0.0, 199.5)] {
        let (dx, dy) = (x - 299.5f64, y - 199.5f64);
        let r = dx.hypot(dy) / unit;
        let ratio = calibration.undistorted_radius(r).unwrap() / r;
        assert!(dx.abs() * ratio < f64::from(corrected.width()) / 2.0);
        assert!(dy.abs() * ratio < f64::from(corrected.height()) / 2.0);
    }
}

#[test]
fn bundled_sony_lensfun_data_covers_a7r_v_and_calibrated_e_mount_lenses() {
    let database: LensDatabase =
        quick_xml::de::from_str(include_str!("../../lensfun_db/mil-sony.xml"))
            .expect("parse bundled Sony Lensfun database");

    let a7r_v = database
        .cameras
        .iter()
        .find(|camera| camera.model.iter().any(|name| name.value == "ILCE-7RM5"));
    let a7r_v = a7r_v.expect("Sony ILCE-7RM5 Lensfun camera entry");
    assert_eq!(a7r_v.mount, "Sony E");
    assert_eq!(a7r_v.cropfactor, 1.0);

    assert!(database.lenses.iter().any(|lens| {
        lens.mount.iter().any(|mount| mount == "Sony E")
            && lens
                .calibration
                .as_ref()
                .is_some_and(|calibration| !calibration.elements.is_empty())
    }));
}

#[test]
fn matches_profiles_for_the_bundled_real_raw_fixtures() {
    let mut sony: LensDatabase =
        quick_xml::de::from_str(include_str!("../../lensfun_db/mil-sony.xml"))
            .expect("parse bundled Sony Lensfun database");
    let mut nikon: LensDatabase =
        quick_xml::de::from_str(include_str!("../../lensfun_db/mil-nikon.xml"))
            .expect("parse bundled Nikon Lensfun database");
    let mut tamron: LensDatabase =
        quick_xml::de::from_str(include_str!("../../lensfun_db/mil-tamron.xml"))
            .expect("parse bundled Tamron Lensfun database");

    sony.lenses.append(&mut nikon.lenses);
    sony.lenses.append(&mut tamron.lenses);

    let sony_match = find_best_lens_match(&sony, "SONY", "FE 20mm F1.8 G")
        .expect("match the Sony a7R V sample lens");
    assert_eq!(sony_match.0, "Sony");
    assert!(sony_match.1.contains("20mm"));
    assert!(
        resolve_lens_params(&sony, &sony_match.0, &sony_match.1, 20.0, Some(8.0), None).is_some()
    );

    let nikon_match = find_best_lens_match(&sony, "NIKON", "NIKKOR Z MC 105mm f/2.8 VR S")
        .expect("match the Nikon Z sample lens");
    assert_eq!(nikon_match.0, "Nikon");
    assert!(nikon_match.1.contains("105mm"));
    assert!(
        resolve_lens_params(
            &sony,
            &nikon_match.0,
            &nikon_match.1,
            105.0,
            Some(4.5),
            None
        )
        .is_some()
    );

    let tamron_match = find_best_lens_match(&sony, "SONY", "E 35-150mm F2.0-F2.8 A058")
        .expect("match the third-party Sony E sample lens");
    assert_eq!(tamron_match.0, "Tamron");
    assert!(
        tamron_match.1.contains("35-150mm"),
        "unexpected Tamron match: {tamron_match:?}"
    );
    assert!(
        resolve_lens_params(
            &sony,
            &tamron_match.0,
            &tamron_match.1,
            45.0,
            Some(4.0),
            None
        )
        .is_some()
    );

    assert!(find_best_lens_match(&sony, "SONY", "Unknown Lens XYZ 999").is_none());
}
