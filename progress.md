# Layered focus stitching — second phase progress

Started from first-phase HEAD `c8b35fe7` on `codex/zhang-haohao-focus-stack`.

Baseline recorded from the first-phase final gates:

- Langyuan TIFF SHA-256: `74942cfea3a1ede229b4449234d62c5429ddd635f7d65f3d6d158f7bc35595d7`
- Wenyuan TIFF SHA-256: `86884740f93c876ee8647c0490fe05215e8bf76a35b33be8a48097798eb855ea`
- Rust test baseline: `646 passed / 0 failed / 28 ignored`
- First-phase reports: `/private/tmp/raw-editor-gate/r045f-final-gates/`

No first-phase diagnostic instrumentation remains in the tracked tree; the only pre-existing worktree changes are `README.md` and `src/components/modals/BatchGeometryModal.tsx`.

## T1 — acceptance harness

Pending.

## T2 — Langyuan 84-image run

Pending; source copy and run must happen after T1 harness validation.

## T3–T7

Pending.

## T1 — acceptance harness (implementation and P94)

- Commit `b99dc773`: replaced the random-report skeleton with a pure ten-condition failure list, strict `DSC_3680.NEF`–`DSC_3763.NEF` enumeration, source snapshots using `source_file_sha256`, default FocusStack invocation, retained report lookup, and the exact npm entry point.
- Commit `64e5ad44`: completed the P94 injection test and corrected the source/group membership condition; P94 passed 100 cases. The 83-file check failed with `missing acceptance sources: ["DSC_3763.NEF"]` in `/private/tmp/raw-editor-gate/r2/t1-missing.log`.

## T2 — Langyuan 84-image run

- Read-only copy: `/private/tmp/raw-editor-stack-smoke/langyuan-84` (84 files, 5,011,254,251 bytes). Report: `/private/tmp/raw-editor-gate/r2/langyuan-84/stack-report-fa2f736a-4186-4b8b-8d06-41e6c1d68e65.json`; run summary: `/private/tmp/raw-editor-gate/r2/langyuan-84.md`.
- The run was rejected after 1,262.71 s because the station graph had 20 disconnected components. It decoded all 84 files, retained unchanged source hashes, grouped 84 frames into 28 stations (6×2, 16×3, 6×4), accepted 8 of 378 relation candidates, and recorded 14 splits. Peak RSS was 12,769,165,312 bytes (2,485 samples) under the 25,769,803,776-byte default threshold. Tone, composition, closure and Quality_Gate were not run after rejection.

## T3–T6

- Commit `604e3288` and follow-up `3e720b8e`: save readback now writes actual format, bit depth, alpha, ICC and path to the matching Stack_Report atomically; TIFF16/PNG16/JPEG and failed-save tests pass.
- Commit `95f1ec23`: production record-mode/default compositor unit coverage. Commit `44de5bed`: production tone P49/P51 tests. Commit `3f494e8d` and `64e5ad44`: production closure P77 and corrected property coverage. P49, P51 and P77 each passed 100 cases in focused runs.
- P89 remains blocked because production has no switchable resident-tile limit; details are in `/private/tmp/raw-editor-gate/r2/t7-p89-blocker.md`.
