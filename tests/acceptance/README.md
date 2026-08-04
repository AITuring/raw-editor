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
