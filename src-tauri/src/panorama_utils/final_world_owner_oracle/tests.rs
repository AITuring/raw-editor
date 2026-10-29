use super::*;

fn sample(source: u64, station: u16, sharpness: f64, cost: f64) -> CandidateSample {
    CandidateSample {
        source,
        station,
        sharpness,
        data_cost: cost,
        seam_cost: 0.0,
        source_x: source as f64 * 10.0,
        source_y: station as f64 * 10.0,
        fully_covered: true,
        flat: true,
        low_frequency_luma: 0.5,
    }
}

fn synthetic(columns: u32, rows: u32, required_regions: usize) -> OracleInput {
    let cells = (0..columns * rows)
        .map(|_| {
            let mut incumbent = sample(2, 1, 0.5, 0.5);
            incumbent.flat = false;
            WorldCell {
                production_source: 2,
                production_station: 1,
                textured_pixels: 1,
                candidates: vec![sample(1, 1, 1.0, 0.0), incumbent],
            }
        })
        .collect();
    OracleInput {
        geometry: WorldGeometry {
            columns,
            rows,
            cell_side_px: 1,
            qg_aligned_region_grid: false,
            world_origin: (100, -200),
            regions: RegionGeometry {
                roi_side_px: 2,
                same_owner_ring_px: 0,
                sampling_phase_px: 1,
            },
        },
        cells,
        required_regions,
        shortfall_limit: 0.05,
    }
}

fn brute_force(input: &OracleInput) -> Option<u64> {
    let options = region_options(input);
    let mut best = None;
    fn visit(
        input: &OracleInput,
        options: &[RegionOption],
        start: usize,
        selected: &mut Vec<usize>,
        best: &mut Option<u64>,
    ) {
        if selected.len() == input.required_regions {
            if let Some(labels) = labels_for(input, selected, options) {
                let cost = labeling_sacrifice(input, &labels);
                *best = Some(best.map_or(cost, |current: u64| current.min(cost)));
            }
            return;
        }
        for index in start..options.len() {
            selected.push(index);
            visit(input, options, index + 1, selected, best);
            selected.pop();
        }
    }
    visit(input, &options, 0, &mut Vec::new(), &mut best);
    best
}

#[test]
fn candidate_origins_come_only_from_coverage_and_content_components() {
    let mut input = synthetic(8, 8, 1);
    for y in 0..8usize {
        for x in 0..8usize {
            if x >= 4 || y >= 4 {
                input.cells[y * 8 + x]
                    .candidates
                    .iter_mut()
                    .find(|candidate| candidate.source == 1)
                    .unwrap()
                    .fully_covered = false;
            }
        }
    }
    let report = solve(&input);
    assert!(report.feasible);
    assert!(
        report
            .selected_regions
            .iter()
            .all(|region| { region.local_origin[0] < 4 && region.local_origin[1] < 4 })
    );
    assert_eq!(report.selected_regions[0].source, 1);
}

#[test]
fn source_order_permutation_world_translation_and_thread_counts_are_deterministic() {
    let input = synthetic(12, 12, 4);
    let expected = solve(&input);
    let mut permuted = input.clone();
    for cell in &mut permuted.cells {
        cell.candidates.reverse();
    }
    assert_eq!(expected, solve(&permuted));

    let mut translated = input.clone();
    translated.geometry.world_origin = (4_100, -7_200);
    let shifted = solve(&translated);
    assert_eq!(expected.labels, shifted.labels);
    assert_eq!(expected.objective_cost, shifted.objective_cost);
    assert!(
        expected
            .selected_regions
            .iter()
            .zip(&shifted.selected_regions)
            .all(|(left, right)| {
                right.world_origin[0] - left.world_origin[0] == 4_000
                    && right.world_origin[1] - left.world_origin[1] == -7_000
            })
    );

    for threads in [1, 2, 8, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let actual = pool.install(|| solve(&input));
        assert_eq!(expected, actual, "{threads} threads");
    }
}

#[test]
fn disconnected_overlap_and_fake_regions_cannot_satisfy_cardinality() {
    let mut input = synthetic(8, 8, 2);
    for (index, cell) in input.cells.iter_mut().enumerate() {
        let (x, y) = (index % 8, index / 8);
        let candidate = cell
            .candidates
            .iter_mut()
            .find(|candidate| candidate.source == 1)
            .unwrap();
        candidate.fully_covered = (x < 2 && y < 2) || (x >= 6 && y >= 6);
    }
    let report = solve(&input);
    assert!(
        report.feasible,
        "two real disconnected components count twice"
    );

    input.required_regions = 3;
    assert!(
        !solve(&input).feasible,
        "duplicates cannot create a third region"
    );

    let mut overlap_only = synthetic(3, 3, 2);
    overlap_only.geometry.regions.sampling_phase_px = 1;
    assert!(
        !solve(&overlap_only).feasible,
        "overlapping windows cannot count twice"
    );
}

#[test]
fn labels_always_retain_one_real_source_and_exact_coordinates() {
    let input = synthetic(8, 8, 1);
    let report = solve(&input);
    let labels = report.labels.unwrap();
    assert!(labels.iter().enumerate().all(|(index, &source)| {
        candidate(&input.cells[index], source)
            .is_some_and(|sample| sample.source_x.is_finite() && sample.source_y.is_finite())
    }));
}

#[test]
fn final_station_overlap_composition_uses_selected_station_but_keeps_all_candidates() {
    let projected = vec![
        ProjectedStationCell {
            final_cell: 0,
            station: 1,
            selected_by_compositor: false,
            samples: vec![sample(11, 1, 1.0, 0.0)],
        },
        ProjectedStationCell {
            final_cell: 0,
            station: 2,
            selected_by_compositor: true,
            samples: vec![sample(22, 2, 0.8, 0.2)],
        },
    ];
    let cells = compose_station_cells(1, &[1], &projected).unwrap();
    assert_eq!(cells[0].production_source, 22);
    assert_eq!(cells[0].production_station, 2);
    assert_eq!(
        cells[0]
            .candidates
            .iter()
            .map(|candidate| candidate.source)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([11, 22])
    );
    let mut invalid = projected.clone();
    invalid[0].selected_by_compositor = true;
    assert!(compose_station_cells(1, &[1], &invalid).is_none());
}

#[test]
fn accepted_seam_cost_rejects_a_sharper_region_that_worsens_strokes() {
    let mut input = synthetic(6, 6, 1);
    for cell in &mut input.cells {
        cell.production_station = 2;
        let current = cell
            .candidates
            .iter_mut()
            .find(|candidate| candidate.source == 2)
            .unwrap();
        current.station = 2;
        current.seam_cost = 0.1;
        let sharper = cell
            .candidates
            .iter_mut()
            .find(|candidate| candidate.source == 1)
            .unwrap();
        sharper.station = 1;
        sharper.seam_cost = 0.2;
    }
    assert!(!solve(&input).feasible);
    for cell in &mut input.cells {
        cell.candidates
            .iter_mut()
            .find(|candidate| candidate.source == 1)
            .unwrap()
            .seam_cost = 0.05;
    }
    assert!(solve(&input).feasible);
}

#[test]
fn small_grid_optimum_and_relaxation_bound_match_independent_bruteforce() {
    for required in 1..=2 {
        let input = synthetic(6, 6, required);
        let options = region_options(&input);
        let brute = brute_force(&input).unwrap();
        let exact = reduced_exact(&input, &options, options.len()).unwrap().0;
        let lower = relaxation_lower_bound(&options, required).unwrap();
        assert_eq!(exact, brute);
        assert!(lower <= brute);
    }
}

#[test]
fn feasible_and_impossible_sharpness_cases_are_reported_without_overstatement() {
    let feasible = solve(&synthetic(12, 12, 4));
    assert!(feasible.feasible);
    assert_eq!(feasible.selected_regions.len(), 4);
    assert!(feasible.feasible_sharpness_coverage.unwrap() >= 0.99);

    let mut impossible = synthetic(12, 12, 4);
    impossible.cells.iter_mut().for_each(|cell| {
        cell.candidates
            .iter_mut()
            .find(|candidate| candidate.source == 1)
            .unwrap()
            .flat = false;
    });
    let report = solve(&impossible);
    assert!(!report.feasible);
    assert!(report.labels.is_none());
}

fn exact_candidate(
    source: u64,
    station_cell: u32,
    sharpness: f32,
    winner: f32,
    shortfall: u8,
) -> ExactCandidateDecision {
    ExactCandidateDecision {
        source,
        station: 1,
        station_cell,
        sharpness_bits: sharpness.to_bits(),
        winner_sharpness_bits: winner.to_bits(),
        shortfall,
        disagreement: 0,
        data_cost_bits: (source as f64).to_bits(),
        pairwise_cost_bits: 0.0f64.to_bits(),
        seam_boundary: false,
    }
}

fn exact_pixel(x: u32, y: u32, station_cell: u32) -> ExactPixelEvidence {
    ExactPixelEvidence {
        x,
        y,
        covered: true,
        textured: true,
        production_source: 2,
        production_station: 1,
        production_shortfall: 10,
        production_disagreement: 0,
        candidates: vec![
            exact_candidate(1, station_cell, 1.0, 1.0, 0),
            exact_candidate(2, station_cell, 0.9, 1.0, 10),
        ],
        source_samples: vec![
            ExactSourceSample {
                source: 1,
                x_bits: f64::from(x).to_bits(),
                y_bits: f64::from(y).to_bits(),
            },
            ExactSourceSample {
                source: 2,
                x_bits: (f64::from(x) + 0.25).to_bits(),
                y_bits: (f64::from(y) + 0.25).to_bits(),
            },
        ],
    }
}

#[test]
fn exact_runs_aggregate_to_bruteforce_owner_metric_for_current_and_arbitrary_labels() {
    let pixels = (0..8).map(|x| exact_pixel(x, 0, 4)).collect::<Vec<_>>();
    let runs = aggregate_exact_runs(8, pixels.clone()).unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].len, 8);
    assert_eq!(runs[0].source_samples.len(), 16);

    let current = exact_run_stats(8, &runs, None, 5);
    assert_eq!(current.textured_pixels, 8);
    assert_eq!(current.shortfall_pixels, 8);
    assert_eq!(current.coverage(), Some(0.0));
    let production =
        super::super::super::stack_pipeline::quality_gate_runner::owner_sharpness_stats(
            &[255; 8], &[1; 8], &[10; 8], &[0; 8], 0,
        )
        .unwrap();
    assert_eq!(current.textured_pixels, production.textured_pixels);
    assert_eq!(current.shortfall_pixels, production.textured_shortfall);

    let labels = vec![1, 2, 1, 2, 1, 2, 1, 2];
    let replay = exact_run_stats(8, &runs, Some(&labels), 5);
    let brute_shortfall = pixels
        .iter()
        .zip(&labels)
        .filter(|(pixel, source)| {
            pixel
                .candidates
                .iter()
                .find(|candidate| candidate.source == **source)
                .is_none_or(|candidate| candidate.shortfall > 5)
        })
        .count() as u64;
    assert_eq!(replay.textured_pixels, 8);
    assert_eq!(replay.shortfall_pixels, brute_shortfall);
    assert_eq!(replay.shortfall_pixels, 4);
}

#[test]
fn exact_runs_split_at_station_transform_coverage_and_quantization_boundaries() {
    let mut pixels = vec![
        exact_pixel(0, 0, 4),
        exact_pixel(1, 0, 4),
        exact_pixel(2, 0, 5),
        exact_pixel(3, 0, 5),
        exact_pixel(4, 0, 5),
    ];
    pixels[3].candidates[1].shortfall = 11;
    pixels[4].covered = false;
    let runs = aggregate_exact_runs(5, pixels).unwrap();
    assert_eq!(
        runs.iter().map(|run| run.len).collect::<Vec<_>>(),
        [2, 1, 1, 1]
    );
    assert_eq!(runs[0].candidates[0].station_cell, 4);
    assert_eq!(runs[1].candidates[0].station_cell, 5);
    assert_eq!(runs[2].candidates[1].shortfall, 11);
    assert!(!runs[3].covered);
}

#[test]
fn exact_runs_preserve_source_provenance_station_seam_and_order_determinism() {
    let mut pixels = (0..4).map(|x| exact_pixel(x, 0, 9)).collect::<Vec<_>>();
    pixels[0].candidates[0].seam_boundary = true;
    pixels[0].candidates[0].pairwise_cost_bits = 0.25f64.to_bits();
    let expected = aggregate_exact_runs(4, pixels.clone()).unwrap();
    assert_eq!(expected[0].len, 1, "seam membership must split a run");
    assert_eq!(expected[0].source_samples[0].source, 1);
    for pixel in &mut pixels {
        pixel.candidates.reverse();
        pixel.source_samples.reverse();
    }
    let permuted = aggregate_exact_runs(4, pixels.clone()).unwrap();
    assert_eq!(expected, permuted);
    for threads in [1, 2, 8, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        assert_eq!(
            expected,
            pool.install(|| aggregate_exact_runs(4, pixels.clone()).unwrap()),
            "{threads} threads"
        );
    }
}

#[test]
fn exact_region_relaxation_never_understates_bruteforce_coverage_bound() {
    let evaluations = (0..4)
        .map(|index| ExactRegionEvaluation {
            proposal: ExactRegionProposal {
                source: 1,
                station: 1,
                left: index * 3,
                top: 0,
                side: 2,
                coarse_current_shortfall_pixels: 2,
            },
            valid: true,
            textured_pixels: 2,
            current_shortfall_pixels: 2,
            baseline_shortfall_pixels: 2,
            candidate_shortfall_pixels: index as u64 % 2,
            owner_change_pixels: index as u64,
            boundary_change_pixels: 0,
            data_cost_delta: index as f64,
            pairwise_cost: 0.0,
            sparse_run_count: 1,
            prediction_replay_mismatch_pixels: 0,
        })
        .collect::<Vec<_>>();
    let report = solve_exact_regions(100, 10, 10, 2, &evaluations);
    assert_eq!(report.selected_shortfall_pixels, Some(6));
    assert_eq!(report.selected_sharpness_coverage, Some(0.94));
    assert!(
        report.relaxation_sharpness_coverage_upper_bound
            >= report.selected_sharpness_coverage.unwrap()
    );

    let mut impossible = evaluations.clone();
    impossible
        .iter_mut()
        .skip(1)
        .for_each(|evaluation| evaluation.valid = false);
    let report = solve_exact_regions(100, 10, 10, 2, &impossible);
    assert!(!report.feasible);
    assert!(report.selected_shortfall_pixels.is_none());
}

#[test]
fn approved_qg_aligned_regions_use_544px_footprints_on_the_fixed_256px_grid() {
    let mut input = synthetic(100, 100, 0);
    input.geometry.cell_side_px = 16;
    input.geometry.qg_aligned_region_grid = true;
    input.geometry.regions = RegionGeometry {
        roi_side_px: 512,
        same_owner_ring_px: 16,
        sampling_phase_px: 256,
    };
    let proposals = exact_region_proposals(&input);
    assert!(!proposals.is_empty());
    assert!(proposals.iter().all(|proposal| proposal.side == 544));
    assert!(
        proposals
            .iter()
            .all(|proposal| { (proposal.left + 16) % 256 == 0 && (proposal.top + 16) % 256 == 0 })
    );
}

#[test]
fn production_geometry_is_544_evidence_and_799_phase_protection() {
    let geometry = RegionGeometry {
        roi_side_px: 512,
        same_owner_ring_px: 16,
        sampling_phase_px: 256,
    };
    assert_eq!(geometry.evidence_span_px(), 544);
    assert_eq!(geometry.protected_span_px(), 799);
}
