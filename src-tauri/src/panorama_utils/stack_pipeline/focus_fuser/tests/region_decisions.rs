//! Region level ownership decisions: content-region candidates and their
//! selection, and the coherent, region and quality-gate decisive cell masks.

use super::*;

#[test]
fn content_regions_come_only_from_current_masks_and_flat_components() {
    let geometry = OwnershipGridGeometry {
        cell_size_px: 100,
        columns: 16,
        rows: 8,
    };
    let count = geometry.cell_count();
    let sharpness = vec![0.02; count];
    let costs = vec![0.1; count];
    let full = ContentEvidence {
        fully_covered: vec![true; count],
        flat: vec![true; count],
        mean_luma: vec![0.5; count],
    };
    let candidates = content_region_candidates(geometry, owner_of(0), &sharpness, &costs, &full);
    assert!(!candidates.is_empty());
    assert!(candidates.iter().all(|candidate| {
        candidate
            .cells
            .iter()
            .all(|&cell| full.fully_covered[cell] && full.flat[cell])
    }));

    let mut masked = full.clone();
    masked.fully_covered[0] = false;
    let masked_candidates =
        content_region_candidates(geometry, owner_of(0), &sharpness, &costs, &masked);
    assert!(
        masked_candidates
            .iter()
            .all(|candidate| !candidate.cells.contains(&0))
    );
    let mut textured = full;
    textured.flat[0] = false;
    let textured_candidates =
        content_region_candidates(geometry, owner_of(0), &sharpness, &costs, &textured);
    assert!(
        textured_candidates
            .iter()
            .all(|candidate| candidate.origin != (0, 0))
    );
}

#[test]
fn content_region_selection_is_permutation_translation_and_thread_deterministic() {
    let geometry = OwnershipGridGeometry {
        cell_size_px: 100,
        columns: 24,
        rows: 8,
    };
    let count = geometry.cell_count();
    let evidence = ContentEvidence {
        fully_covered: vec![true; count],
        flat: vec![true; count],
        mean_luma: vec![0.4; count],
    };
    let candidates = content_region_candidates(
        geometry,
        owner_of(0),
        &vec![0.01; count],
        &vec![0.2; count],
        &evidence,
    );
    let expected = select_content_regions(candidates.clone(), 2);
    let mut reversed = candidates.clone();
    reversed.reverse();
    assert_eq!(expected, select_content_regions(reversed, 2));
    let threaded = std::thread::spawn(move || select_content_regions(candidates, 2))
        .join()
        .expect("content selector thread");
    assert_eq!(expected, threaded);

    let shifted_geometry = OwnershipGridGeometry {
        columns: 32,
        ..geometry
    };
    let shifted_count = shifted_geometry.cell_count();
    let mut shifted = ContentEvidence {
        fully_covered: vec![false; shifted_count],
        flat: vec![false; shifted_count],
        mean_luma: vec![f64::NAN; shifted_count],
    };
    for row in 0..8usize {
        for column in 8..32usize {
            let cell = row * 32 + column;
            shifted.fully_covered[cell] = true;
            shifted.flat[cell] = true;
            shifted.mean_luma[cell] = 0.4;
        }
    }
    let translated = select_content_regions(
        content_region_candidates(
            shifted_geometry,
            owner_of(0),
            &vec![0.01; shifted_count],
            &vec![0.2; shifted_count],
            &shifted,
        ),
        2,
    );
    assert_eq!(translated.len(), expected.len());
    assert!(
        translated
            .iter()
            .zip(&expected)
            .all(|(translated, original)| translated.origin.0 == original.origin.0 + 8)
    );
}

#[test]
fn disconnected_and_overlapping_content_regions_cannot_count_twice() {
    let geometry = OwnershipGridGeometry {
        cell_size_px: 100,
        columns: 8,
        rows: 8,
    };
    let count = geometry.cell_count();
    let mut evidence = ContentEvidence {
        fully_covered: vec![true; count],
        flat: vec![true; count],
        mean_luma: vec![0.5; count],
    };
    for row in 0..8usize {
        evidence.flat[row * 8 + 4] = false;
    }
    assert!(
        content_region_candidates(
            geometry,
            owner_of(0),
            &vec![0.01; count],
            &vec![0.1; count],
            &evidence,
        )
        .is_empty()
    );

    let duplicate = ContentRegionCandidate {
        owner: owner_of(0),
        cells: (0..64).collect(),
        origin: (0, 0),
        data_cost: 0.1,
    };
    let mut overlap = duplicate.clone();
    overlap.owner = owner_of(1);
    overlap.origin = (1, 0);
    assert_eq!(select_content_regions(vec![duplicate, overlap], 2).len(), 1);
}

#[test]
fn decisive_sharpness_requires_a_coherent_non_vetoed_component() {
    let geometry = OwnershipGridGeometry {
        cell_size_px: 8,
        columns: 4,
        rows: 2,
    };
    let mut candidate = CandidateCells::new(8);
    candidate.covered.fill(true);
    candidate.sharpness.fill(0.5);
    for cell in [0usize, 1, 4, 5] {
        candidate.sharpness[cell] = 0.9;
    }
    candidate.sharpness[7] = 0.9;
    let decisive = coherent_decisive_cells(geometry, &vec![0.5; 8], &candidate);
    assert_eq!(
        decisive,
        vec![true, true, false, false, true, true, false, false]
    );

    candidate.disagreement[0] = OWNERSHIP_DISAGREEMENT_VETO + 0.01;
    let vetoed = coherent_decisive_cells(geometry, &vec![0.5; 8], &candidate);
    assert!(vetoed.iter().all(|&value| !value));

    candidate.sharpness.fill(0.1);
    let base = coherent_base_cells(geometry, &vec![0.5; 8], &candidate);
    assert!(base.iter().all(|&value| value));
    candidate.sharpness[7] = 0.49;
    let still_coherent = coherent_base_cells(geometry, &vec![0.5; 8], &candidate);
    assert!(!still_coherent[7]);
    assert!(still_coherent[..7].iter().all(|&value| value));
}

#[test]
fn region_decision_keeps_a_quality_roi_sized_owner_block_together() {
    let geometry = OwnershipGridGeometry {
        cell_size_px: 16,
        columns: 50,
        rows: 50,
    };
    let count = geometry.cell_count();
    let mut candidate = CandidateCells::new(count);
    candidate.covered.fill(true);
    candidate.sharpness.fill(0.9);
    let decisive = region_decisive_cells(geometry, &vec![0.5; count], &candidate);
    assert!(decisive.iter().all(|&value| value));

    candidate.disagreement[0] = OWNERSHIP_DISAGREEMENT_VETO + 0.01;
    let vetoed = region_decisive_cells(geometry, &vec![0.5; count], &candidate);
    assert!(vetoed.iter().all(|&value| !value));
}

#[test]
fn quality_region_accepts_a_near_best_full_roi_without_relaxing_shortfall() {
    let geometry = OwnershipGridGeometry {
        cell_size_px: 16,
        columns: 40,
        rows: 40,
    };
    let count = geometry.cell_count();
    let mut candidate = CandidateCells::new(count);
    candidate.covered.fill(true);
    candidate.sharpness.fill(0.951);
    let owner = vec![0.94; count];
    let best = vec![1.0; count];
    let decisive = quality_region_decisive_cells(geometry, &owner, &best, &candidate);
    assert!(decisive.iter().all(|&value| value));

    // 0.94 is a six-percent shortfall from the observed winner, so the
    // same region is rejected at the unchanged five-percent gate boundary.
    candidate.sharpness.fill(0.94);
    let rejected = quality_region_decisive_cells(geometry, &owner, &best, &candidate);
    assert!(rejected.iter().all(|&value| !value));

    // A near-winner is still not allowed to lower an already sharper
    // owner just because it makes a larger connected ROI.  This is the
    // regression that let the smoothness override create gate shortfall.
    candidate.sharpness.fill(0.93);
    let softer = quality_region_decisive_cells(geometry, &vec![0.94; count], &best, &candidate);
    assert!(softer.iter().all(|&value| !value));
}

#[test]
fn quality_gate_decisive_cells_only_moves_a_consistent_five_percent_winner() {
    let geometry = OwnershipGridGeometry {
        cell_size_px: 16,
        columns: 2,
        rows: 2,
    };
    let count = geometry.cell_count();
    let mut candidate = CandidateCells::new(count);
    candidate.covered.fill(true);
    candidate.sharpness.fill(0.96);
    let owner = vec![0.90; count];
    let best = vec![0.96; count];
    let decisive = quality_gate_decisive_cells(geometry, &owner, &best, &candidate);
    assert!(decisive.iter().all(|&value| value));

    // A candidate that is sharper than the current owner but six percent
    // below the observed winner cannot manufacture a passing owner.
    candidate.sharpness.fill(0.90);
    let rejected = quality_gate_decisive_cells(geometry, &owner, &best, &candidate);
    assert!(rejected.iter().all(|&value| !value));

    candidate.sharpness.fill(0.96);
    candidate
        .disagreement
        .fill(OWNERSHIP_DISAGREEMENT_VETO + 0.01);
    let vetoed = quality_gate_decisive_cells(geometry, &owner, &best, &candidate);
    assert!(vetoed.iter().all(|&value| !value));
}

#[test]
fn content_constraint_keeps_real_provenance_without_forcing_an_oracle() {
    let geometry = OwnershipGridGeometry {
        cell_size_px: 100,
        columns: 24,
        rows: 8,
    };
    let count = geometry.cell_count();
    let evidence = ContentEvidence {
        fully_covered: vec![true; count],
        flat: vec![true; count],
        mean_luma: vec![0.5; count],
    };
    let mut candidates = content_region_candidates(
        geometry,
        owner_of(0),
        &vec![0.2; count],
        &vec![0.4; count],
        &evidence,
    );
    candidates.extend(content_region_candidates(
        geometry,
        owner_of(1),
        &vec![0.3; count],
        &vec![0.1; count],
        &evidence,
    ));
    let selected = select_content_regions(candidates.clone(), 2);
    assert_eq!(selected.len(), 2);
    assert!(selected.iter().all(|region| region.owner == owner_of(1)));
    assert!(selected.iter().all(|region| candidates.contains(region)));
}

#[test]
fn content_constraint_handles_feasible_and_impossible_cardinality() {
    let geometry = OwnershipGridGeometry {
        cell_size_px: 100,
        columns: 80,
        rows: 8,
    };
    let count = geometry.cell_count();
    let sharpness = vec![0.01; count];
    let costs = vec![0.1; count];
    let feasible = content_region_candidates(
        geometry,
        owner_of(0),
        &sharpness,
        &costs,
        &ContentEvidence {
            fully_covered: vec![true; count],
            flat: vec![true; count],
            mean_luma: vec![0.5; count],
        },
    );
    assert_eq!(
        select_content_regions(feasible, CONTENT_OWNER_REGION_TARGET).len(),
        CONTENT_OWNER_REGION_TARGET
    );

    let impossible = content_region_candidates(
        geometry,
        owner_of(0),
        &sharpness,
        &costs,
        &ContentEvidence {
            fully_covered: vec![true; count],
            flat: vec![false; count],
            mean_luma: vec![0.5; count],
        },
    );
    assert!(impossible.is_empty());
}
