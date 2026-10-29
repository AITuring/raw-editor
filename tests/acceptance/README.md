# Acceptance fixtures

This directory defines the fixture contract used by the regular regression suite and opt-in local
real-RAW acceptance runs. The automated tests generate deterministic RGB and noise patterns in
memory, so the repository does not need to carry derived previews or exports.

Real RAW files are intentionally external because of their size and licensing. Add an entry to
`raw-manifest.json` before placing a file in a local fixture directory. Each entry must record the
camera, capture purpose, original filename, byte size, SHA-256, source, license or owner permission,
and expected sensor dimensions. Never commit a RAW file without explicit redistribution permission.

The Sony α7R V baseline is now partially established by `DSC05363.ARW`: a real 60 MP, ISO 125,
25-second exposure in Sony's lossy-compressed ARW mode. It covers the `lossy-arw` scenario and supports
scoped pipeline-quality inspection plus local release-performance measurements. Daylight, tungsten,
high-ISO, underexposed, saturated-light, lossless-L and uncompressed cases remain deferred.
The measured result and visual findings are recorded in
[`sony-a7rv-60mp-baseline.md`](sony-a7rv-60mp-baseline.md).

This single file has no color target or paired ACR rendering, so its quality conclusion is limited to
detecting visible orientation, unpacking, demosaic, white-balance, color-transform, preview and export
defects. It does not establish absolute color accuracy or Adobe Camera Raw parity. The separate
ILCE-7M4 (α7 IV) and Nikon Z 6 fixtures remain cross-camera development coverage and must not be
reported as α7R V results.

Regular CI covers:

- rawler's built-in Sony ILCE-7RM5 camera profile and color matrices;
- the bundled Lensfun database and α7R V camera entry;
- deterministic local denoise behavior;
- full-dimension export and embedded ICC round trips;
- versioned preview ROI framing, bounds clamping and memory-budgeted image caches;
- sidecar v0-to-v1 migration, unknown-field preservation and crash recovery;
- batch-export result aggregation where one failed item does not discard later results.

Large local acceptance runs may use `RAW_EDITOR_ACCEPTANCE_DIR` as a read-only fixture root. Tests must
write generated output to a temporary directory, never beside source photographs.

The focus-stack image sets under `/Users/dp/Downloads/书画拼图` are catalogued in
[`focus-stack-datasets.json`](focus-stack-datasets.json). `兰亭` is the mandatory visual baseline: its
45 source JPEGs exclude all derived `FocusStack` and Photoshop comparison files, and its existing
`DSC08854_FocusStack.jpg` must keep its recorded dimensions, byte size, and SHA-256. The other folders
are staged in increasing cost order for local replay; derived Photoshop files are reference material and
are never treated as production inputs.

Run the fast local dataset contract after every code change:

```sh
RAW_EDITOR_FOCUS_STACK_DATA_ROOT="/Users/dp/Downloads/书画拼图" \
  npm run focus-stack-datasets:check -- --require-data
```

Add `--hash-inputs` when a stage changes source preparation, caching, decoding, or coordinate handling;
this verifies the frozen aggregate digest for the catalogued source sets. The contract never writes beside
the source images and never replaces the 兰亭 reference output.

For any production behavior change, run the mandatory 兰亭 protection regression with the dataset runner.
It writes only to `/private/tmp/raw-editor-regression` (override with `--out-dir`); the reference JPEG is
read-only and its manifest digest is re-checked before and after the run.

```sh
RAW_EDITOR_FOCUS_STACK_DATA_ROOT="/Users/dp/Downloads/书画拼图" \
  npm run focus-stack-regression -- --dataset lanting --label <stage>
```

The runner executes the ignored `real_focus_stack_fixture_from_env` harness (default 10% render,
preview-only; `--scale 1` writes the full TIFF/app JPEG) and fails when an input is missing from the
stitching order or from the streaming compositor, when per-pixel source ownership verification
disappears, or when the reference file changes. It compares the preview and six crop digests with
[`focus-stack-regression-baselines.json`](focus-stack-regression-baselines.json): an identical digest
proves a refactor kept the output. When the output changed, the registered sharpness floors apply: the
harness registers the output onto the reference rendering with FAST/BRIEF + RANSAC and compares the
contrast-normalised gradient energy (`mean |∇I|² / var I`) of every content tile, overall and per
left/center/right third, against the recorded baseline. A changed output that passes the floors still
needs a visual check of the preview and crops. Other datasets run the same way (`--dataset mayuan-psd`,
`--subset langyuan-10`, `--limit 10`); datasets without a reference report only output-side sharpness.
`--binary <test executable>` runs a prebuilt `cargo test --release --no-run` binary, and
`--update-baseline` or `--record-summary <run>.summary.json` records a verified run, and `--rebaseline`
records an intended, visually reviewed change that deviates from the old baseline (crashes, missing inputs and
reference changes still fail). The baseline keeps only counts, digests and metric summaries.

The equivalent manual command is below. Do not add `--exact` to the Rust filter because the test is
nested under `panorama_stitching::acceptance_tests::focus_stack_fixture`. `RAW_EDITOR_FOCUS_STACK_REFERENCE_JPEG` requests a
byte-for-byte comparison with the Photoshop file; the canvas differs from Photoshop's, so leave it unset
for protection runs.

```sh
LANTING_PATHS=$(find "/Users/dp/Downloads/书画拼图/兰亭" -maxdepth 1 -type f \
  -iname 'DSC*.jpg' ! -iname '*FocusStack*' -print | sort | paste -sd: -)
RAW_EDITOR_FOCUS_STACK_PATHS="$LANTING_PATHS" \
RAW_EDITOR_STACK_ACCEPTANCE_OUTPUT="/private/tmp/raw-editor-lanting-baseline.tiff" \
RAW_EDITOR_STACK_REPORT_DIR="/private/tmp/raw-editor-lanting-stack-reports" \
RAW_EDITOR_STACK_ACCEPTANCE_METRICS="/private/tmp/raw-editor-lanting-baseline.metrics.json" \
RAW_EDITOR_STACK_ACCEPTANCE_METRICS_REFERENCE="/Users/dp/Downloads/书画拼图/兰亭/DSC08854_FocusStack.jpg" \
RAW_EDITOR_STACK_ACCEPTANCE_CROPS=1 CARGO_NET_OFFLINE=true \
cargo test --release --manifest-path src-tauri/Cargo.toml --lib \
  real_focus_stack_fixture_from_env -- --ignored --nocapture --test-threads=1
```

Run the real-RAW acceptance test explicitly:

```sh
RAW_EDITOR_ACCEPTANCE_DIR="$PWD/src/assets/test" \
RAW_EDITOR_ACCEPTANCE_REPORT="/private/tmp/raw-editor-real-raw-report.json" \
RAW_EDITOR_ACCEPTANCE_OUTPUT_DIR="/private/tmp/raw-editor-real-raw-output" \
cargo test --release --manifest-path src-tauri/Cargo.toml \
  local_real_raw_pipeline_acceptance -- --ignored --nocapture --test-threads=1
```

The ignored test verifies the byte size and SHA-256 before decoding. It then checks the decoded camera
identity, full sensor mosaic dimensions, source and unpacked-container bit depths, CFA, black/white
levels, as-shot white balance and color matrices. A centered, CFA-aligned crop of the real mosaic
isolates rescaling, quality demosaic, neutral-white-balance color calibration and
as-shot-white-balance color calibration. Finally, the production RAW developer creates the full
oriented image; the CPU display path derives a 1920-pixel preview and the production encoder creates
an ICC-tagged full-resolution JPEG. Stage hashes and timings are diagnostic observations and are not
pass/fail performance thresholds.
