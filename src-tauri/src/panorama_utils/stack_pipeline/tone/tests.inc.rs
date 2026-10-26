#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fewer_than_1024_samples_keep_identity() {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        let samples = vec![
            ToneSample {
                owner: [0.5; 3],
                source: [0.4; 3],
            };
            TONE_MIN_SAMPLES - 1
        ];
        let solve = solve_tone_pair(&samples);
        assert_eq!(solve.status, ToneSolveStatus::Identity);
        assert_eq!(solve.gain, [1.0; 3]);
        assert_eq!(solve.offset, [0.0; 3]);
    }

    #[test]
    fn pair_solve_rejects_outliers_and_keeps_the_median_gain() {
        let mut samples = Vec::with_capacity(TONE_MIN_SAMPLES + 64);
        for index in 0..TONE_MIN_SAMPLES + 64 {
            let source = 0.1 + (index % 31) as f32 / 40.0;
            samples.push(ToneSample {
                owner: [source * 1.1, source * 0.95, source * 1.05],
                source: [source; 3],
            });
        }
        for sample in &mut samples[..64] {
            sample.owner = [0.95; 3];
        }
        let solve = solve_tone_pair(&samples);
        assert_eq!(solve.status, ToneSolveStatus::Applied);
        for (channel, expected) in [1.1f32, 0.95, 1.05].into_iter().enumerate() {
            assert!((solve.gain[channel] - expected).abs() < 1e-3, "{solve:?}");
            assert!(
                (solve.solved_gain[channel] - expected).abs() < 1e-3,
                "{solve:?}"
            );
            assert!(solve.offset[channel].abs() < 1e-3, "{solve:?}");
        }
        assert!(!solve.gain_clamped);
        // Every consistent sample is kept; the painted-over ones are not.
        assert_eq!(solve.retained_samples, TONE_MIN_SAMPLES);
    }

    #[test]
    fn pair_solve_is_not_flattened_where_the_fields_disagree_by_position() {
        // Two exposures of the same content, each with its own ±6% low
        // frequency deviation (vignetting, a moved highlight). The true
        // relation is identity; a least-squares affine fit of one field on the
        // other attenuates its slope to ≈0.96 and compensates with an offset,
        // the station seam this solve used to create.
        let samples = (0..4096)
            .map(|index| {
                let content = 0.25 + 0.2 * ((index * 37) % 101) as f32 / 100.0;
                let owner = 1.0 + 0.06 * ((index as f32) * 0.71).sin();
                let source = 1.0 + 0.06 * ((index as f32) * 1.37 + 0.4).cos();
                ToneSample {
                    owner: [content * owner; 3],
                    source: [content * source; 3],
                }
            })
            .collect::<Vec<_>>();
        let solve = solve_tone_pair(&samples);
        assert_eq!(solve.status, ToneSolveStatus::Applied);
        for channel in 0..3 {
            assert!((solve.gain[channel] - 1.0).abs() < 0.01, "{solve:?}");
            assert!(solve.offset[channel].abs() < 0.005, "{solve:?}");
        }
    }

    #[test]
    fn identity_tone_preserves_owner_pixels_outside_evidence() {
        let source = Rgb32FImage::from_pixel(32, 32, Rgb([0.2, 0.3, 0.4]));
        let owner = Rgb32FImage::from_pixel(32, 32, Rgb([0.7, 0.6, 0.5]));
        let evidence = GrayImage::new(32, 32);
        let result = apply_low_frequency_tone(&source, &owner, &evidence, &ToneSolve::identity(0));
        assert_eq!(result, owner);
    }

    #[test]
    fn delta_e00_is_symmetric_and_zero_for_identical_rgb() {
        let left = [0.23, 0.41, 0.67];
        let right = [0.71, 0.36, 0.19];
        assert!(delta_e00_rgb(left, left).abs() < 1e-12);
        assert!((delta_e00_rgb(left, right) - delta_e00_rgb(right, left)).abs() < 1e-12);
    }

    fn textured(x: f64, y: f64) -> f32 {
        (0.45 + 0.12 * (x * 0.013 + y * 0.007).sin() + 0.08 * (x * 0.004 - y * 0.011).cos()) as f32
    }

    /// A tile whose pixel `(x, y)` shows world `(x + origin_x, y)`, scaled by
    /// `gain`, with the columns `uncovered` left as transparent zeros.
    fn shifted_tile(
        width: u32,
        height: u32,
        origin_x: f64,
        gain: f32,
        uncovered: std::ops::Range<u32>,
    ) -> Rgb32FImage {
        Rgb32FImage::from_fn(width, height, |x, y| {
            if uncovered.contains(&x) {
                Rgb([0.0; 3])
            } else {
                Rgb([gain * textured(f64::from(x) + origin_x, f64::from(y)); 3])
            }
        })
    }

    #[test]
    fn world_aligned_tone_field_reads_covered_pixels_only() {
        // Left 64 columns uncovered; the covered part is a constant 0.5.
        let tile =
            Rgb32FImage::from_fn(
                256,
                128,
                |x, _| {
                    if x < 64 { Rgb([0.0; 3]) } else { Rgb([0.5; 3]) }
                },
            );
        let field = world_aligned_tone_tile(3, 7, &tile, (100, 40, 355, 167), |x, y| {
            Some((x - 100.0, y - 40.0))
        });
        assert_eq!((field.low.width(), field.low.height()), (16, 8));
        assert_eq!(field.world_origin, (100.0, 40.0));
        for y in 0..8 {
            for x in 0..16 {
                let level = field.validity.get_pixel(x, y)[0];
                if x < 4 {
                    assert_eq!(level, 0, "uncovered cell ({x}, {y})");
                } else {
                    assert_eq!(level, TONE_CELL_SAMPLE, "covered cell ({x}, {y})");
                    // Next to the uncovered edge the field is not darkened.
                    for channel in field.low.get_pixel(x, y).0 {
                        assert!((channel - 0.5).abs() < 1e-5, "({x}, {y}) {channel}");
                    }
                }
            }
        }
        // A cell straddling the coverage edge is corrected but no evidence.
        let straddling =
            Rgb32FImage::from_fn(
                64,
                16,
                |x, _| {
                    if x < 40 { Rgb([0.0; 3]) } else { Rgb([0.5; 3]) }
                },
            );
        let field = world_aligned_tone_tile(0, 1, &straddling, (0, 0, 63, 15), |x, y| Some((x, y)));
        let levels = (0..4)
            .map(|x| field.validity.get_pixel(x, 0)[0])
            .collect::<Vec<_>>();
        assert_eq!(levels, vec![0, 0, TONE_CELL_PARTIAL, TONE_CELL_SAMPLE]);
    }

    #[test]
    fn world_aligned_tone_field_follows_the_world_to_tile_map() {
        // The tile is drawn rotated by 90°: world (x, y) shows tile (y, 255 - x).
        let tile = Rgb32FImage::from_fn(256, 256, |x, y| {
            Rgb([textured(f64::from(x), f64::from(y)); 3])
        });
        let field =
            world_aligned_tone_tile(0, 1, &tile, (0, 0, 255, 255), |x, y| Some((y, 255.0 - x)));
        let axis = world_aligned_tone_tile(0, 1, &tile, (0, 0, 255, 255), |x, y| Some((x, y)));
        // World cell (cx, cy) of the rotated field is axis cell (cy, 15 - cx).
        for (cx, cy) in [(2u32, 3u32), (7, 12), (11, 5)] {
            let rotated = field.low.get_pixel(cx, cy)[0];
            let expected = axis.low.get_pixel(cy, 15 - cx)[0];
            assert!(
                (rotated - expected).abs() < 2e-3,
                "({cx}, {cy}) {rotated} vs {expected}"
            );
        }
    }

    /// Samples of one overlap where `first` shows the content at `first_gain`
    /// and `second` at `second_gain`.
    fn relation_between(
        first: usize,
        second: usize,
        first_gain: f32,
        second_gain: f32,
    ) -> ToneRelation {
        let samples = (0..2048)
            .map(|index| {
                let content = 0.2 + 0.5 * ((index * 53) % 97) as f32 / 96.0;
                ToneSample {
                    owner: [content * first_gain; 3],
                    source: [content * second_gain; 3],
                }
            })
            .collect::<Vec<_>>();
        ToneRelation::new(first, second, consistent_samples(&samples))
    }

    #[test]
    fn joint_solve_equalises_a_loop_without_an_anchor_or_order() {
        // Three stations exposed at 1.0, 0.9 and 1.1 with all three overlaps
        // accepted: every corrected tile shows the content at one level, the
        // component keeps zero mean log gain, and the relation order is
        // irrelevant.
        let exposure = [1.0f32, 0.9, 1.1];
        let relations = [(0, 1), (1, 2), (0, 2)].map(|(first, second)| {
            relation_between(first, second, exposure[first], exposure[second])
        });
        let transforms = solve_tone_transforms(3, &relations);
        let level = |tile: usize| transforms[tile].gain[0] * exposure[tile];
        let gains = transforms
            .iter()
            .map(|transform| transform.gain)
            .collect::<Vec<_>>();
        for tile in 1..3 {
            assert!((level(tile) - level(0)).abs() < 1e-4, "{gains:?}");
        }
        let mean_log_gain = transforms
            .iter()
            .map(|transform| f64::from(transform.gain[0]).ln())
            .sum::<f64>()
            / 3.0;
        assert!(mean_log_gain.abs() < 1e-6);
        assert!(
            transforms
                .iter()
                .all(|transform| transform.supported && !transform.clamped)
        );
        assert!(
            transforms
                .iter()
                .all(|transform| transform.offset[0].abs() < 1e-4)
        );

        let reversed = [(0, 2), (1, 2), (0, 1)].map(|(first, second)| {
            relation_between(first, second, exposure[first], exposure[second])
        });
        let reordered = solve_tone_transforms(3, &reversed);
        for (left, right) in transforms.iter().zip(&reordered) {
            assert!((left.gain[0] - right.gain[0]).abs() < 1e-6);
        }
    }

    #[test]
    fn joint_solve_bounds_each_tile_and_keeps_the_solved_values() {
        // A 1.6× step exceeds the 1.25 gain bound on one side.
        let relations = [relation_between(0, 1, 1.0, 0.4)];
        let transforms = solve_tone_transforms(3, &relations);
        let corrected = &transforms[1];
        assert!(corrected.clamped);
        assert_eq!(corrected.gain, [TONE_MAX_GAIN; 3]);
        assert!((corrected.solved_gain[0] - (2.5f32).sqrt()).abs() < 1e-3);
        assert!(
            corrected
                .offset
                .iter()
                .all(|value| value.abs() <= TONE_MAX_ABS_OFFSET)
        );
        // A tile without an accepted relation stays exact identity.
        assert!(!transforms[2].supported);
        assert_eq!(transforms[2].gain, [1.0; 3]);
        assert_eq!(transforms[2].offset, [0.0; 3]);
    }

    #[test]
    fn harmonized_tiles_match_a_darker_neighbour_without_uncovered_bias() {
        // `harmonize_tone_tiles` writes the process-wide run ledger.
        let _run_scope = degradation::begin_run_scope();
        // Tile 0 covers world x ∈ [0, 1024); tile 1 covers [512, 1408) and is
        // 10% darker. Its first 128 columns (world 384..512, inside tile 0)
        // are uncovered zeros that must not bias the pair relation.
        let (width, height) = (1_408u32, 640u32);
        let left = shifted_tile(1_024, height, 0.0, 1.0, 0..0);
        let right = shifted_tile(1_024, height, 384.0, 0.9, 0..128);
        let tiles = [
            world_aligned_tone_tile(10, 1, &left, (0, 0, 1_023, height - 1), |x, y| Some((x, y))),
            world_aligned_tone_tile(11, 2, &right, (384, 0, width - 1, height - 1), |x, y| {
                Some((x - 384.0, y))
            }),
        ];
        let owners = (0..height)
            .flat_map(|_| (0..width).map(|x| if x < 768 { 1u16 } else { 2 }))
            .collect::<Vec<_>>();
        let mut panorama = Rgb32FImage::from_fn(width, height, |x, y| {
            if x < 768 {
                *left.get_pixel(x, y)
            } else {
                *right.get_pixel(x - 384, y)
            }
        });
        let evidence = GrayImage::from_pixel(width, height, Luma([255]));
        let report =
            harmonize_tone_tiles(&mut panorama, &owners, width, &tiles, &evidence, &[(0, 1)]);

        // Zero mean log gain: the darker tile is raised and the brighter one
        // lowered by the same factor, √(1/0.9) each.
        let (brighter, darker) = (&report.tiles[0], &report.tiles[1]);
        assert_eq!((brighter.station_index, darker.station_index), (10, 11));
        for channel in 0..3 {
            assert!(!darker.clamped && !brighter.clamped, "{report:?}");
            let step = darker.gain[channel] / brighter.gain[channel];
            assert!((step - 1.0 / 0.9).abs() < 0.01, "{report:?}");
            assert!((darker.gain[channel] * brighter.gain[channel] - 1.0).abs() < 1e-3);
            assert!(darker.offset[channel].abs() < 0.005, "{report:?}");
        }
        assert!(report.pairs_without_evidence.is_empty());
        assert_eq!(report.status, ToneStatus::Applied);
        // The corrected right half continues the corrected left half's tone.
        let level = brighter.gain[0] as f32;
        for (x, y) in [(100u32, 100u32), (700, 300), (900, 300), (1_200, 450)] {
            let expected = level * textured(f64::from(x), f64::from(y));
            let value = panorama.get_pixel(x, y)[0];
            assert!(
                (value - expected).abs() < 0.01,
                "({x}, {y}) {value} vs {expected}"
            );
        }
        assert!(report.boundary_delta_e.max <= 1.5, "{report:?}");
    }

    #[test]
    fn only_accepted_station_relations_are_tone_pairs() {
        let _run_scope = degradation::begin_run_scope();
        // Two overlapping tiles 20% apart, but no accepted Station_Relation
        // between them: their overlap is no tone evidence (需求 9.1).
        let (width, height) = (1_536u32, 640u32);
        let left = shifted_tile(1_024, height, 0.0, 1.0, 0..0);
        let right = shifted_tile(1_024, height, 512.0, 0.8, 0..0);
        let tiles = [
            world_aligned_tone_tile(0, 1, &left, (0, 0, 1_023, height - 1), |x, y| Some((x, y))),
            world_aligned_tone_tile(1, 2, &right, (512, 0, width - 1, height - 1), |x, y| {
                Some((x - 512.0, y))
            }),
        ];
        let owners = (0..height)
            .flat_map(|_| (0..width).map(|x| if x < 768 { 1u16 } else { 2 }))
            .collect::<Vec<_>>();
        let mut panorama = Rgb32FImage::from_fn(width, height, |x, y| {
            if x < 768 {
                *left.get_pixel(x, y)
            } else {
                *right.get_pixel(x - 512, y)
            }
        });
        let before = panorama.clone();
        let evidence = GrayImage::from_pixel(width, height, Luma([255]));
        let report = harmonize_tone_tiles(&mut panorama, &owners, width, &tiles, &evidence, &[]);
        assert_eq!(report.status, ToneStatus::Identity);
        assert!(report.pairs_without_evidence.is_empty());
        assert!(report.tiles.iter().all(|tile| tile.gain == [1.0; 3]));
        assert_eq!(panorama, before);

        // The same pair as an accepted relation is corrected.
        let report =
            harmonize_tone_tiles(&mut panorama, &owners, width, &tiles, &evidence, &[(1, 0)]);
        assert_eq!(report.status, ToneStatus::Applied, "{report:?}");
        assert!((report.tiles[1].gain[0] / report.tiles[0].gain[0] - 1.25).abs() < 0.01);
    }

    #[test]
    fn an_accepted_relation_without_a_covered_overlap_is_recorded_without_evidence() {
        let _run_scope = degradation::begin_run_scope();
        let left = shifted_tile(256, 256, 0.0, 1.0, 0..0);
        // The bounding boxes intersect on world x ∈ [192, 256), but tile 1 is
        // uncovered there: no doubly covered cell and so no retained sample.
        let right = shifted_tile(256, 256, 192.0, 0.8, 0..64);
        let tiles = [
            world_aligned_tone_tile(0, 1, &left, (0, 0, 255, 255), |x, y| Some((x, y))),
            world_aligned_tone_tile(1, 2, &right, (192, 0, 447, 255), |x, y| {
                Some((x - 192.0, y))
            }),
        ];
        let owners = (0..256u32)
            .flat_map(|_| (0..448u32).map(|x| if x < 256 { 1u16 } else { 2 }))
            .collect::<Vec<_>>();
        let mut panorama = Rgb32FImage::new(448, 256);
        let before = panorama.clone();
        let evidence = GrayImage::from_pixel(448, 256, Luma([255]));
        let report =
            harmonize_tone_tiles(&mut panorama, &owners, 448, &tiles, &evidence, &[(0, 1)]);
        assert_eq!(report.status, ToneStatus::Identity);
        assert_eq!(
            report.pairs_without_evidence,
            vec![TonePairWithoutEvidenceRecord {
                left: 0,
                right: 1,
                retained_samples: 0,
            }]
        );
        assert!(
            report
                .tiles
                .iter()
                .all(|tile| tile.gain == [1.0; 3] && !tile.clamped)
        );
        assert_eq!(panorama, before);
    }

    #[test]
    fn applied_correction_is_continuous_across_analysis_cells() {
        let _run_scope = degradation::begin_run_scope();
        // A 30% step on a strong horizontal gradient. The correction is
        // `(gain − 1) · low + offset`; read from the analysis grid by nearest
        // cell it would step by ≈1e-3 every 16 px, bilinear it changes per
        // pixel only by the gradient's own share (≈7e-5).
        let (width, height) = (1_536u32, 640u32);
        let ramp = |x: f64| (0.1 + 0.8 * x / 1_535.0) as f32;
        let left = Rgb32FImage::from_fn(1_024, height, |x, _| Rgb([ramp(f64::from(x)); 3]));
        let right = Rgb32FImage::from_fn(1_024, height, |x, _| {
            Rgb([0.7 * ramp(f64::from(x) + 512.0); 3])
        });
        let tiles = [
            world_aligned_tone_tile(0, 1, &left, (0, 0, 1_023, height - 1), |x, y| Some((x, y))),
            world_aligned_tone_tile(1, 2, &right, (512, 0, width - 1, height - 1), |x, y| {
                Some((x - 512.0, y))
            }),
        ];
        let owners = (0..height)
            .flat_map(|_| (0..width).map(|x| if x < 768 { 1u16 } else { 2 }))
            .collect::<Vec<_>>();
        let mut panorama = Rgb32FImage::from_fn(width, height, |x, y| {
            if x < 768 {
                *left.get_pixel(x, y)
            } else {
                *right.get_pixel(x - 512, y)
            }
        });
        let before = panorama.clone();
        let evidence = GrayImage::from_pixel(width, height, Luma([255]));
        let report =
            harmonize_tone_tiles(&mut panorama, &owners, width, &tiles, &evidence, &[(0, 1)]);
        assert_eq!(report.status, ToneStatus::Applied, "{report:?}");
        assert!(report.tiles.iter().all(|tile| !tile.clamped), "{report:?}");
        let y = height / 2;
        let delta = |x: u32| panorama.get_pixel(x, y)[0] - before.get_pixel(x, y)[0];
        for x in 0..width - 1 {
            // The owner boundary is where the correction is meant to change.
            if x + 1 == 768 {
                continue;
            }
            let step = (delta(x + 1) - delta(x)).abs();
            assert!(step < 3e-4, "correction steps by {step} at x = {x}");
        }
        // Across the owner boundary the corrected tiles meet.
        let seam = panorama.get_pixel(768, y)[0] - panorama.get_pixel(767, y)[0];
        assert!(seam.abs() < 3e-3, "seam step {seam}");
    }

    /// Colour scene with broad tone variation and fine texture.
    fn scan_scene(x: f64, y: f64) -> [f32; 3] {
        let broad = textured(x, y);
        let fine: f32 = if ((x as i64 / 3) + (y as i64 / 3)) % 2 == 0 {
            0.03
        } else {
            -0.03
        };
        [
            broad + fine,
            0.9 * broad + 0.03 + fine,
            0.75 * broad + 0.08 + fine,
        ]
    }

    // 检查点 12: on a synthetic scan grid with arbitrary per-station RGB
    // exposure differences, every Owner_Region boundary band meets
    // Delta_E00 ≤ 1.5 after harmonisation (需求 9.8) and the correction
    // carries no high frequency, i.e. every output pixel keeps its owner's
    // high-frequency residual (需求 9.2).
    #[test]
    fn synthetic_scan_grid_with_exposure_steps_meets_the_boundary_bound() {
        let _run_scope = degradation::begin_run_scope();
        let (columns, rows) = (3usize, 2usize);
        let (tile_width, tile_height, step) = (1_024u32, 1_024u32, 512u32);
        let width = step * (columns as u32 - 1) + tile_width;
        let height = step * (rows as u32 - 1) + tile_height;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for _case in 0..3 {
            let stations = columns * rows;
            let origins = (0..stations)
                .map(|station| {
                    (
                        (station % columns) as u32 * step,
                        (station / columns) as u32 * step,
                    )
                })
                .collect::<Vec<_>>();
            // Independent RGB exposures in [0.88, 1.12] per station.
            let exposures = (0..stations)
                .map(|_| std::array::from_fn::<f32, 3, _>(|_| (0.88 + 0.24 * next()) as f32))
                .collect::<Vec<_>>();
            let sources = (0..stations)
                .map(|station| {
                    let (ox, oy) = origins[station];
                    Rgb32FImage::from_fn(tile_width, tile_height, |x, y| {
                        let scene = scan_scene(f64::from(x + ox), f64::from(y + oy));
                        Rgb(std::array::from_fn(|channel| {
                            scene[channel] * exposures[station][channel]
                        }))
                    })
                })
                .collect::<Vec<_>>();
            let tiles = (0..stations)
                .map(|station| {
                    let (ox, oy) = origins[station];
                    world_aligned_tone_tile(
                        station,
                        station as u16 + 1,
                        &sources[station],
                        (ox, oy, ox + tile_width - 1, oy + tile_height - 1),
                        |x, y| Some((x - f64::from(ox), y - f64::from(oy))),
                    )
                })
                .collect::<Vec<_>>();
            // Hard ownership by the nearest station centre.
            let owner_at = |x: u32, y: u32| {
                (0..stations)
                    .filter(|&station| {
                        let (ox, oy) = origins[station];
                        (ox..ox + tile_width).contains(&x) && (oy..oy + tile_height).contains(&y)
                    })
                    .min_by_key(|&station| {
                        let (ox, oy) = origins[station];
                        let dx = i64::from(x) - i64::from(ox + tile_width / 2);
                        let dy = i64::from(y) - i64::from(oy + tile_height / 2);
                        (dx * dx + dy * dy, station)
                    })
                    .expect("the grid covers the canvas")
            };
            let owners = (0..height)
                .flat_map(|y| (0..width).map(move |x| (x, y)))
                .map(|(x, y)| owner_at(x, y) as u16 + 1)
                .collect::<Vec<_>>();
            let mut panorama = Rgb32FImage::from_fn(width, height, |x, y| {
                let station = owners[(y * width + x) as usize] as usize - 1;
                let (ox, oy) = origins[station];
                *sources[station].get_pixel(x - ox, y - oy)
            });
            let before = panorama.clone();
            // Accepted relations between grid neighbours only.
            let relations = (0..stations)
                .flat_map(|station| {
                    let right = (station % columns + 1 < columns).then_some((station, station + 1));
                    let below =
                        (station + columns < stations).then_some((station, station + columns));
                    [right, below].into_iter().flatten()
                })
                .collect::<Vec<_>>();
            let evidence = GrayImage::from_pixel(width, height, Luma([255]));
            let report =
                harmonize_tone_tiles(&mut panorama, &owners, width, &tiles, &evidence, &relations);
            assert_eq!(
                report.status,
                ToneStatus::Applied,
                "{exposures:?} {report:?}"
            );
            assert!(
                report.boundary_delta_e.max <= 1.5,
                "{exposures:?} {report:?}"
            );
            assert!(report.tiles.iter().all(|tile| !tile.clamped), "{report:?}");
            // The correction is a low-frequency field inside each Owner_Region:
            // neighbouring pixels of one owner change by the same amount up to
            // the field's own slope, so the fine checker survives exactly.
            let delta = |x: u32, y: u32, channel: usize| {
                panorama.get_pixel(x, y)[channel] - before.get_pixel(x, y)[channel]
            };
            for y in (0..height).step_by(7) {
                for x in 0..width - 1 {
                    let index = (y * width + x) as usize;
                    if owners[index] != owners[index + 1] {
                        continue;
                    }
                    for channel in 0..3 {
                        let step = (delta(x + 1, y, channel) - delta(x, y, channel)).abs();
                        assert!(step < 1e-3, "({x}, {y}) channel {channel} steps by {step}");
                    }
                }
            }
        }
    }

    #[test]
    fn constant_gray_samples_have_a_finite_affine_solution() {
        let samples = vec![
            ToneSample {
                owner: [0.44; 3],
                source: [0.40; 3],
            };
            TONE_MIN_SAMPLES
        ];
        let solve = solve_tone_pair(&samples);
        assert!(solve.gain.iter().all(|value| value.is_finite()));
        assert!(solve.offset.iter().all(|value| value.is_finite()));
        for channel in 0..3 {
            let corrected = solve.gain[channel] * 0.40 + solve.offset[channel];
            assert!((corrected - 0.44).abs() < 1e-4);
        }
    }
}
