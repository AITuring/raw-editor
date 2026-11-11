//! Exact effective coverage in the physical Source_RAW domain.
//!
//! The denominator follows the immutable native inverse maps, never alpha,
//! ownership, luminance or the success of a rendered pixel. A calibrated
//! canvas rectangle includes lens padding and is not a sensor footprint.
use super::super::quality_gate::EffectivePixelCoverage;
use super::super::report::EffectiveSourceDomainRecord;
use super::{QualityGateInput, degradation};
use image::GrayImage;
use nalgebra::{Point2, Point3};
use rayon::prelude::*;

pub(super) fn applicable(input: &QualityGateInput<'_>) -> bool {
    input.sources.iter().any(|source| {
        source
            .native_registration
            .as_ref()
            .and_then(|native| native.refinement.source_sampling.as_ref())
            .is_some_and(|sampling| sampling.has_calibrated_footprint())
    })
}

pub(super) fn measure(
    input: &QualityGateInput<'_>,
    legacy: &EffectivePixelCoverage,
) -> Result<(EffectivePixelCoverage, EffectiveSourceDomainRecord), &'static str> {
    // Fail closed for malformed maps, rather than interpreting mapping errors
    // as absent source area and accidentally shrinking the denominator.
    for source in input.sources {
        let native = source
            .native_registration
            .as_ref()
            .ok_or(degradation::QUALITY_GATE_INSUFFICIENT_EVIDENCE)?;
        if native.refinement.source_sampling.is_none()
            || !native.world_to_station.iter().all(|v| v.is_finite())
            || native.world_to_station.determinant().abs() < 1e-12
            || !native.station_origin.0.is_finite()
            || !native.station_origin.1.is_finite()
        {
            return Err(degradation::DIAGNOSTICS_ROI_INVALID);
        }
    }
    let counts = count_domain(input.coverage, input.world_origin, |world| {
        input.sources.iter().any(|source| {
            let native = source
                .native_registration
                .as_ref()
                .expect("validated native map");
            let unwarped = input
                .residual
                .warp_inverse(world, source.geometry.station_id);
            let Some(station) = native.corrected_station(unwarped) else {
                return false;
            };
            native
                .refinement
                .source_sampling
                .as_ref()
                .expect("validated sampling")
                .source_point(Point3::new(
                    station.x + native.station_origin.0,
                    station.y + native.station_origin.1,
                    1.0,
                ))
                .is_some()
        })
    })?;
    let result = EffectivePixelCoverage {
        output_nontransparent: counts.covered_inside,
        projected_union: counts.domain,
        ratio: counts.covered_inside as f64 / counts.domain as f64,
    };
    let record = EffectiveSourceDomainRecord {
        method: "native_inverse_sensor_domain_pixel_centres_v1".into(),
        source_count: input.sources.len(),
        legacy_rectangular_union: legacy.projected_union,
        all_output_nontransparent: legacy.output_nontransparent,
        covered_inside: counts.covered_inside,
        missing_inside: counts.domain - counts.covered_inside,
        covered_outside: counts.covered_outside,
    };
    Ok((result, record))
}

#[derive(Default, Debug, PartialEq)]
struct Counts {
    domain: u64,
    covered_inside: u64,
    covered_outside: u64,
}

fn count_domain(
    coverage: &GrayImage,
    origin: (f64, f64),
    supported: impl Fn(Point2<f64>) -> bool + Sync,
) -> Result<Counts, &'static str> {
    if coverage.width() == 0
        || coverage.height() == 0
        || !origin.0.is_finite()
        || !origin.1.is_finite()
    {
        return Err(degradation::DIAGNOSTICS_ROI_INVALID);
    }
    // Rayon reduces integers only. No full-canvas support bitmap or RAW decode.
    let counts = coverage
        .as_raw()
        .par_chunks(coverage.width() as usize)
        .enumerate()
        .map(|(y, row)| {
            let mut counts = Counts::default();
            for (x, &alpha) in row.iter().enumerate() {
                let inside = supported(Point2::new(
                    origin.0 + x as f64 + 0.5,
                    origin.1 + y as f64 + 0.5,
                ));
                counts.domain += u64::from(inside);
                counts.covered_inside += u64::from(inside && alpha != 0);
                counts.covered_outside += u64::from(!inside && alpha != 0);
            }
            counts
        })
        .reduce(Counts::default, |a, b| Counts {
            domain: a.domain + b.domain,
            covered_inside: a.covered_inside + b.covered_inside,
            covered_outside: a.covered_outside + b.covered_outside,
        });
    if counts.domain == 0 {
        return Err(degradation::QUALITY_GATE_INSUFFICIENT_EVIDENCE);
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn sensor_domain_is_independent_of_alpha_and_retains_real_holes() {
        let full = GrayImage::from_pixel(100, 100, Luma([255]));
        let mut holes = full.clone();
        for y in 20..40 {
            for x in 20..40 {
                holes.put_pixel(x, y, Luma([0]));
            }
        }
        // Nonzero origin and curved lens-like border; no image intensity input.
        let support =
            |p: Point2<f64>| (p.x - 151.0).powi(2) + (p.y + 19.0).powi(2) < 49.0_f64.powi(2);
        let a = count_domain(&full, (101.0, -69.0), support).unwrap();
        let b = count_domain(&holes, (101.0, -69.0), support).unwrap();
        let empty = count_domain(&GrayImage::new(100, 100), (101.0, -69.0), support).unwrap();
        assert_eq!(a.domain, b.domain);
        assert_eq!(a.domain, empty.domain);
        assert_eq!(empty.covered_inside, 0);
        assert_eq!(a.covered_inside - b.covered_inside, 400);
        assert!(b.covered_inside as f64 / (b.domain as f64) < 0.98);
        assert!(a.covered_outside > 0);
        assert_eq!(a.covered_inside + a.covered_outside, 10_000);
    }

    #[test]
    fn absent_or_nonfinite_sensor_domain_cannot_pass() {
        let alpha = GrayImage::from_pixel(10, 10, Luma([255]));
        assert!(count_domain(&alpha, (0.0, 0.0), |_| false).is_err());
        assert!(count_domain(&alpha, (f64::NAN, 0.0), |_| true).is_err());
    }
    #[test]
    fn native_domain_uses_refinement_origin_and_black_source_geometry() {
        use super::super::{NativeReferenceMap, QualitySource};
        use crate::panorama_utils::stack_pipeline::{
            quality_gate::SourceGeometry, residual_warp::ResidualWarp,
        };
        use crate::panorama_utils::{
            shared::ImageInfo,
            stitching::{FocusSourceRefinement, FocusSourceSampling},
        };
        use nalgebra::Matrix3;
        let image = ImageInfo {
            id: 0,
            filename: "black.nef".into(),
            width: 10,
            height: 10,
            alignment_image: GrayImage::new(10, 10),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm: None,
            source_footprint: None,
            overview_reference: false,
            features: vec![],
            top_features: vec![],
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: vec![],
            vertical_edge_columns: vec![],
        };
        let translation = Matrix3::new(1.0, 0.0, 100.0, 0.0, 1.0, -50.0, 0.0, 0.0, 1.0);
        let sampling = FocusSourceSampling::for_test(
            &image,
            &translation,
            crate::panorama_utils::coordinates::Projection::Planar,
            None,
        );
        let source = QualitySource {
            owner: 1,
            path: "black.nef".into(),
            dimensions: (10, 10),
            geometry: SourceGeometry {
                member_to_anchor: Matrix3::identity(),
                tile_to_world: translation,
                station_id: 0,
            },
            native_warp: None,
            native_registration: Some(NativeReferenceMap {
                mosaic_tone: None,
                source_prefilter_divisor: 1.0,
                world_to_station: translation.try_inverse().unwrap(),
                station_to_world: translation,
                station_origin: (100.0, -50.0),
                refinement: FocusSourceRefinement {
                    translation: (2, 0),
                    source_sampling: Some(sampling),
                    ..Default::default()
                },
            }),
        };
        let sources = [source];
        let output = image::Rgb32FImage::new(14, 10);
        let coverage = GrayImage::from_pixel(14, 10, Luma([255]));
        let residual = ResidualWarp::new();
        let input = QualityGateInput {
            output: &output,
            coverage: &coverage,
            ownership: &[],
            confidence: &[],
            textured: None,
            owner_shortfall: None,
            owner_disagreement: None,
            unresolved_pixel_count: 0,
            world_origin: (100.0, -50.0),
            sources: &sources,
            accepted_station_relations: &[],
            residual: &residual,
            acceptance_render_scale: 1.0,
            final_sharpen_amount: 0.0,
        };
        let legacy = EffectivePixelCoverage {
            output_nontransparent: 140,
            projected_union: 140,
            ratio: 1.0,
        };
        let (result, audit) = measure(&input, &legacy).unwrap();
        assert_eq!(result.projected_union, 100);
        assert_eq!(result.output_nontransparent, 100);
        assert_eq!(audit.covered_outside, 40);
        assert_eq!(audit.missing_inside, 0);
        // Uncalibrated paths retain the legacy result unless explicitly tested here.
        assert!(!applicable(&input));
    }
}
