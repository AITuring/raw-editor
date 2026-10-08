# Code Summary

## Implemented

- Restored the final Station Relation scale policy and its independent Node/Rust oracles to the inclusive `[0.95, 1.05]` interval. Prior-repair scale bounds continue to reuse these constants; Station Grouper `[0.98, 1.02]`, search candidates, local-scale, tone, and anisotropic measurement fixtures were left unchanged.
- Added typed `CanonicalRelationView` / `CanonicalRelationViewError`. Both the final solver and VirtualTile authoritative-cluster filter now use the same content-derived direction, point orientation, dimensions, and canonical homography. Out-of-range indices are invariant failures; singular transforms are recoverable rejections.
- Added pair-level, serialized-report-safe accounting for non-invertible canonical transforms with stable reason `station_relation_non_invertible_canonical_transform` and stage `canonical_transform_non_invertible`. Repeated candidates/topology fallback do not double count, and fitted singular candidates remain `unmeasured=0`.
- Strengthened Property 94 and `stack_acceptance_failures()` to require 28 indexed stations, no undecodable sources, an independently connected accepted-relation graph, exact 84-member connectivity, and exact 28-station GraphCut fusion coverage.
- Added acceptance directory preflight validation for absolute/non-nested paths and 84 readable, hashable, regular non-symlink files. Stack Report selection now requires exactly one regular `stack-report-*.json` and ignores unrelated JSON.
- Corrected final VirtualTile connectivity diagnostics to publish original SourceRaw paths rather than `virtual://` tile identifiers.
- Added boundary, reversed-direction, inverse-scale, singular-transform, acceptance-invariant, path-validation, report-selection, and connectivity-member regressions.
- Updated `.kiro/specs/layered-camera-group-focus-stitching/tasks.md` and `docs/focus-stack-reference-validation.md` with the latest valid 84-image result.
- Checked all 12 plan items in `04-plan.md`.

## Design Deviations

None. The approved Rust/Tauri/nalgebra/serde/sha2/proptest/Node stack was retained, no dependency was added, no network access or server/watcher was used, and no acceptance threshold was relaxed. The latest report did not justify a speculative algorithm change: measured fitted relations outside `[0.95,1.05]` remain rejected.

The workflow-level instruction required a commit, which superseded the plan document's earlier no-commit execution note. Existing user edits in `README.md` and `src/components/modals/BatchGeometryModal.tsx` were preserved and excluded from the task commit.

## Latest 84-Image Acceptance

Latest valid run: `.agents/artifacts/approved-scale-and-84-acceptance-2026-10-08/20261008T084725Z/`.

- Evidence integrity: one parseable Stack Report; before/after HEAD both `e44c681be95446e08141bf25f2dc85af50a1ba6e`; before/after relevant diff SHA-256 both `8a35eb6faba806563944493d69f4e92c66f0a9faaefb6ab585f2da8ee2b6cd0e`; report SHA-256 `dc13a47788b6eca432ea19936f382c310705594d242441a8dcba3369e57be9c6`.
- Inputs/grouping: 84/84 decoded and digest-stable; 28 stations; 84 unique members; isolated=0; undecodable=0.
- Relations: candidates=378, accepted=8, rejected=370, unmeasured=316, prior repairs=0. Failure stages: fitted=32, no_predicted_overlap=194, insufficient_measurable_patches=95, insufficient_polished_support=22, insufficient_bidirectional_matches=20, insufficient_grid_coverage=7. Reasons: scale_out_of_range=24, edge_orientation_mismatch=2.
- Connectivity: 20 components; component member counts `[3,3,4,4,3,4,7,6,3,3,7,3,4,6,4,4,4,6,3,3]`, totaling 84 unique original SourceRaw paths.
- Pipeline stopped before fusion/composition/tone/residual-warp/Quality Gate. Fusion count=0, composition union/opaque=0/0, Quality Gate=`not_run`; these are not passes.
- Resources: duration 1,082,125 ms; memory threshold 25,769,803,776 bytes; peak RSS 11,212,750,848 bytes; 2,134 samples; network requests=0.
- Result: algorithmic rejection (`geometry_disconnected`, harness exit 101). Fitted scales included 0.7091, 0.94645, 1.06884, and 1.53428; prior repairs also exceeded approved shift/scale bounds. The next step is a product/evidence decision about whether real adjacent stations may vary by more than 5%, or an additional independent measurement that distinguishes incorrect scale fits from real focal-distance changes.

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
- Full ordered gate `npm run focus-stack-quality:check && npm run image-stack-preview:check && npm run typecheck && cargo fmt ... --check && cargo test ... --lib && git diff --check` — pre-mapping: 673 passed, 0 failed, 28 ignored; post-mapping: 674 passed, 0 failed, 28 ignored. Node focus/preview checks and TypeScript typecheck passed; formatting and diff checks passed.
- Final `cargo test ... --lib` — `674 passed; 0 failed; 28 ignored; finished in 277.92s`.
- Final `git diff --check && npm run focus-stack-quality:check` — passed.
- First acceptance run (`20261008T082000Z`) — unique valid report, code/RAW evidence stable, exit 101 with 20 components; exposed the `virtual://` member-report mapping defect.
- Final acceptance run (`20261008T084725Z`) — unique valid report, code/RAW evidence stable, original source members correctly reported, exit 101 with the same 20-component algorithmic result.
