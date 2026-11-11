use super::*;
use image::Luma;
use image::Rgb32FImage;
use nalgebra::Matrix3;

fn cache_source(path: &str) -> QualitySource {
    QualitySource {
        owner: 1,
        path: path.to_string(),
        geometry: SourceGeometry {
            member_to_anchor: Matrix3::identity(),
            tile_to_world: Matrix3::identity(),
            station_id: 0,
        },
        dimensions: (16, 16),
        native_warp: None,
        native_registration: None,
    }
}

fn cache_roi(x: u32) -> super::super::quality_gate::QualityRoi {
    super::super::quality_gate::QualityRoi {
        x,
        y: 0,
        side: 2,
        region_label: x + 1,
        owner: 1,
        world_origin: (f64::from(x), 0.0),
        boundary_clearance: 4.0,
    }
}

#[test]
fn quality_roi_cache_reuses_entries_and_separates_source_identity() {
    let source = cache_source("a.nef");
    let other_source = cache_source("b.nef");
    let roi = cache_roi(0);
    let mut cache = QualityRoiCache::new(2 * 2 * 3 * std::mem::size_of::<f32>());
    cache.insert(&source, &roi, Rgb32FImage::new(2, 2));
    assert!(cache.take(&source, &roi).is_some());
    assert!(cache.take(&other_source, &roi).is_none());
    assert_eq!(
        cache.stats().hits,
        1,
        "the source ROI should be a cache hit"
    );
    assert_eq!(cache.stats().misses, 1, "a different source must miss");
}

#[test]
fn quality_roi_cache_evicts_oldest_entry_at_byte_limit() {
    let source = cache_source("a.nef");
    let first = cache_roi(0);
    let second = cache_roi(1);
    let bytes = 2 * 2 * 3 * std::mem::size_of::<f32>();
    let mut cache = QualityRoiCache::new(bytes);
    cache.insert(&source, &first, Rgb32FImage::new(2, 2));
    cache.insert(&source, &second, Rgb32FImage::new(2, 2));
    assert!(cache.take(&source, &first).is_none());
    assert!(cache.take(&source, &second).is_some());
    assert_eq!(
        cache.stats().evictions,
        1,
        "the oldest ROI should be evicted"
    );
    assert_eq!(cache.stats().peak_bytes, bytes);
}

#[test]
fn quality_roi_cache_hot_cycles_keep_lru_metadata_bounded() {
    let source = cache_source("hot.nef");
    let roi = cache_roi(0);
    let bytes = 2 * 2 * 3 * std::mem::size_of::<f32>();
    let mut cache = QualityRoiCache::new(bytes);
    cache.insert(&source, &roi, Rgb32FImage::new(2, 2));
    for _ in 0..1000 {
        let reference = cache.take(&source, &roi).unwrap();
        assert!(cache.order.len() <= cache.entries.len());
        cache.insert(&source, &roi, reference);
        assert_eq!(cache.order.len(), cache.entries.len());
    }
    assert_eq!(cache.stats().hits, 1000);
    assert_eq!(cache.order.len(), 1);
}

#[test]
fn quality_roi_cache_invalidates_when_interior_owner_or_reference_mapping_changes() {
    let _scope = degradation::begin_run_scope();
    crate::panorama_utils::stack_pipeline::tone::reset_run_records();
    let output = Rgb32FImage::new(16, 16);
    let coverage = GrayImage::from_pixel(16, 16, Luma([255]));
    let confidence = vec![0.8; 256];
    let residual = ResidualWarp::new();
    let mut ownership = vec![1; 256];
    let mut sources = [cache_source("origin.nef"), cache_source("inside.nef")];
    sources[1].owner = 2;
    sources[1].native_registration = Some(NativeReferenceMap {
        mosaic_tone: None,
        source_prefilter_divisor: 1.0,
        world_to_station: Matrix3::identity(),
        station_to_world: Matrix3::identity(),
        station_origin: (0.0, 0.0),
        refinement: Default::default(),
    });
    let bind = |cache: &mut QualityRoiCache, ownership: &[u16], sources: &[QualitySource]| {
        cache.bind_input(&QualityGateInput {
            output: &output,
            coverage: &coverage,
            ownership,
            confidence: &confidence,
            textured: None,
            owner_shortfall: None,
            owner_disagreement: None,
            unresolved_pixel_count: 0,
            world_origin: (0.0, 0.0),
            sources,
            accepted_station_relations: &[],
            residual: &residual,
            acceptance_render_scale: 1.0,
            final_sharpen_amount: 0.0,
        });
    };
    let mut roi = cache_roi(0);
    roi.region_label = quality_gate::station_level_roi_label(0).unwrap();
    let mut cache = QualityRoiCache::new(1024);
    bind(&mut cache, &ownership, &sources);
    cache.insert(
        &sources[0],
        &roi,
        Rgb32FImage::from_pixel(2, 2, Rgb([0.1; 3])),
    );
    bind(&mut cache, &ownership, &sources);
    let unchanged = cache
        .take(&sources[0], &roi)
        .expect("unchanged second pass must hit");
    cache.insert(&sources[0], &roi, unchanged);
    ownership[17] = 2; // Interior (1,1) changes; origin owner stays one.
    assert_eq!(ownership[0], 1);
    bind(&mut cache, &ownership, &sources);
    assert!(cache.take(&sources[0], &roi).is_none());
    cache.insert(&sources[0], &roi, Rgb32FImage::new(2, 2));
    sources[1]
        .native_registration
        .as_mut()
        .unwrap()
        .refinement
        .translation = (1, -2);
    bind(&mut cache, &ownership, &sources);
    assert!(
        cache.take(&sources[0], &roi).is_none(),
        "an interior owner's registration change invalidates origin-owner key"
    );
    cache.insert(&sources[0], &roi, Rgb32FImage::new(2, 2));
    sources[1].geometry.tile_to_world[(0, 2)] = 0.25;
    bind(&mut cache, &ownership, &sources);
    assert!(cache.take(&sources[0], &roi).is_none());
    assert_eq!(cache.stats().hits, 1);
    assert_eq!(cache.stats().misses, 3);
}

mod boundary_alignment;
mod coverage_evidence;
mod criterion_rois;
mod gate_outcomes;
mod reference_sampling;
mod tone_owner_fidelity;
