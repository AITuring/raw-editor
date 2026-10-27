# Code Summary

## Implemented

- Restored the final Station Relation scale policy and its independent Node/Rust oracles to the inclusive `[0.95, 1.05]` interval. Prior-repair scale bounds continue to reuse these constants; Station Grouper `[0.98, 1.02]`, search candidates, local-scale, tone, and anisotropic measurement fixtures were left unchanged.
- Added typed `CanonicalRelationView` / `CanonicalRelationViewError` and a complete typed Station Relation measurement candidate. Both the final solver and VirtualTile authoritative-cluster filter now share content-derived direction, point orientation, dimensions, canonical homography, low-frequency mean, edge-strength, edge orientation, convexity, and the same rejection gate. Out-of-range indices are invariant failures; singular transforms are recoverable rejections.
- Added pair-level, serialized-report-safe accounting for non-invertible canonical transforms with stable reason `station_relation_non_invertible_canonical_transform` and stage `canonical_transform_non_invertible`. Repeated candidates/topology fallback do not double count, and fitted singular candidates remain `unmeasured=0`.
- Strengthened Property 94 and `stack_acceptance_failures()` to require 28 indexed stations, no undecodable grouping entries, every `input.sources[*].decoded` flag to be true, an independently connected accepted-relation graph, exact 84-member connectivity, and exact 28-station GraphCut fusion coverage.
- Added write-free acceptance directory containment resolution for nonexistent report paths and their nearest existing parent; rejected nested preflights leave the source inventory unchanged. Stack Report selection still requires exactly one regular `stack-report-*.json`.
- Corrected final VirtualTile connectivity diagnostics to publish original SourceRaw paths rather than `virtual://` tile identifiers.
- Added boundary, reversed-direction, inverse-scale, singular-transform, acceptance-invariant, path-validation, report-selection, and connectivity-member regressions.
- Updated `.kiro/specs/layered-camera-group-focus-stitching/tasks.md` and `docs/focus-stack-reference-validation.md` with the latest valid 84-image result.
- Checked all 12 plan items in `04-plan.md`.

## Design Deviations

None. The approved Rust/Tauri/nalgebra/serde/sha2/proptest/Node stack was retained, no dependency was added, no network access or server/watcher was used, and no acceptance threshold was relaxed. The latest report did not justify a speculative algorithm change: measured fitted relations outside `[0.95,1.05]` remain rejected.

The workflow's accidental commit `3771f1ba` was left untouched as required. No commit, amend, reset, rebase, or push was performed. Existing user edits in `README.md` and `src/components/modals/BatchGeometryModal.tsx` remain preserved and were not edited by this review-fix task. Four reviewed raw artifacts were removed only from the current workspace snapshot and are now ignored; their blobs remain in the old commit history pending separate user authorization for history cleanup.

## Latest 84-Image Acceptance

Latest valid review-fix run: `.agents/artifacts/approved-scale-and-84-acceptance-2026-10-08/20261008T093406Z/` (local ignored artifact).

- Evidence integrity: one parseable Stack Report; 84/84 source digests stable; report SHA-256 `eaa54e6400c0924553152eaa2f61cde6176639828600ee4250dd05e7e8ffc114`; network requests=0. Command: `CARGO_NET_OFFLINE=true npm run stack-acceptance:check` with the frozen absolute source/report paths.
- Inputs/grouping: 84/84 decoded and digest-stable; 28 stations; 84 unique members; isolated=0; undecodable=0.
- Relations: candidates=378, accepted=8, rejected=370, unmeasured=316, prior repairs=0. Failure stages: fitted=32, no_predicted_overlap=194, insufficient_measurable_patches=95, insufficient_polished_support=22, insufficient_bidirectional_matches=20, insufficient_grid_coverage=7. Reasons: scale_out_of_range=24, edge_orientation_mismatch=2.
- Connectivity: 20 components; component member counts `[3,3,4,4,3,4,7,6,3,3,7,3,4,6,4,4,4,6,3,3]`, totaling 84 unique original SourceRaw paths.
- Prior-free path: 74 candidate records, including 31 fitted; planned repairs for stations 7/15/16/26/27 were all rejected by existing bounds (scales 0.8270/0.8997/0.8270/0.7989/0.7291, shifts 151.3–1315.5 px), so applied repairs=0.
- Pipeline stopped before final fusion/composition/tone/residual-warp/Quality Gate. Fusion count=0, composition union/opaque=0/0, Quality Gate=`not_run`; station-local GraphCut logs are not final-stage entry.
- Resources: elapsed 1,111.71 seconds; memory threshold 25,769,803,776 bytes; peak RSS 10,656,382,976 bytes; 2,189 samples; network requests=0.
- Result: algorithmic rejection (`geometry_disconnected`, harness exit 101). No threshold was relaxed. The next 84-image P0 is to persist and verify tile/native/world transform decomposition for the five out-of-bounds planned proposals, then use that evidence to correct any coordinate-scale bug before addressing no-predicted-overlap, measurable support, and polish edge loss.

## Tests Run

All Cargo-backed commands used `CARGO_NET_OFFLINE=true` and direct Cargo tests used `--offline --locked`.

- `npm run focus-stack-quality:check` — passed repeatedly; final output: `Validated focus-stack quality guards ... 66 stable failure identifiers ...`.
- `cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check` — initially reported formatting diffs in the newly edited files; after targeted `rustfmt --edition 2024` it passed on every rerun.
- `cargo test ... --lib canonical_relation_view -- --nocapture` — 2 passed, 0 failed.
- `cargo test ... --lib station_relation_contract -- --nocapture` — 2 passed, 0 failed.
- `cargo test ... --lib non_invertible_canonical_transform -- --nocapture` — 2 passed, 0 failed.
- `cargo test ... --lib authoritative -- --nocapture` — 1 passed, 0 failed.
- `cargo test ... --lib property_94_stack_acceptance_verdict_is_pure -- --nocapture` — one parallel attempt was interrupted with exit `-1` while contending for Cargo locks; immediate isolated rerun passed: 1 passed, 0 failed.
- `cargo test ... --lib validate_acceptance_directories -- --nocapture` — 1 passed, 0 failed.
- `cargo test ... --lib stack_report_selection -- --nocapture` — 1 passed, 0 failed.
- `cargo test ... --lib property_29_station_relation_acceptance_is_all_gates -- --nocapture` — 1 passed, 0 failed.
- `cargo test ... --lib property_30_station_relation_defects_map_to_exact_reasons -- --nocapture` — 1 passed, 0 failed.
- `cargo test ... --lib prior_repair_bounds_use_finite_inclusive_shift_and_scale_limits -- --nocapture` — 1 passed, 0 failed.
- `cargo test ... --lib focus_capture_group_solver_rejects_out_of_contract_station_scale -- --nocapture` — 1 passed, 0 failed.
- `cargo test ... --lib acceptance_ -- --nocapture` — 6 passed, 0 failed, 10 ignored before the report mapping fix; 6 passed, 0 failed, 10 ignored after it.
- `cargo test ... --lib virtual_tile -- --nocapture` — 77 passed, 0 failed.
- `cargo test ... --lib virtual_tile_connectivity_reports_original_source_members -- --nocapture` — 1 passed, 0 failed.
- Full ordered gate `npm run focus-stack-quality:check`, `npm run image-stack-preview:check`, `npm run typecheck`, Rust fmt check, full offline locked Rust tests, and `git diff --check` — passed. Rust result: 676 passed, 0 failed, 28 ignored.
- Final `cargo test ... --lib --offline --locked` — `676 passed; 0 failed; 28 ignored; finished in 340.15s`.
- Final `git diff --check && npm run focus-stack-quality:check` — passed.
- First acceptance run (`20261008T082000Z`) — unique valid report, code/RAW evidence stable, exit 101 with 20 components; exposed the `virtual://` member-report mapping defect.
- Review-fix acceptance run (`20261008T093406Z`) — unique valid ignored report, code/RAW evidence stable, exit 101 with unchanged 20-component algorithmic result; report and raw log are intentionally not tracked.
