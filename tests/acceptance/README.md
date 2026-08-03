# Acceptance fixtures

This directory defines the small, redistributable fixtures used by the regular regression suite.
The automated tests generate deterministic RGB and noise patterns in memory, so the repository does
not need to carry derived previews or exports.

Real RAW files are intentionally external because of their size and licensing. Add an entry to
`raw-manifest.json` before placing a file in a local fixture directory. Each entry must record the
camera, capture purpose, original filename, byte size, SHA-256, source, license or owner permission,
and expected sensor dimensions. Never commit a RAW file without explicit redistribution permission.

The Sony α7R V image-quality/performance baseline remains deferred. Its manifest entries are reserved
here so a later run can reproduce daylight, tungsten, high-ISO, underexposed, saturated-light and each
ARW compression-mode case without changing the test contract.

Regular CI covers:

- rawler's built-in Sony ILCE-7RM5 camera profile and color matrices;
- the bundled Lensfun database and α7R V camera entry;
- deterministic local denoise behavior;
- full-dimension export and embedded ICC round trips;
- sidecar v0-to-v1 migration, unknown-field preservation and crash recovery;
- batch-export result aggregation where one failed item does not discard later results.

Large local acceptance runs may use `RAW_EDITOR_ACCEPTANCE_DIR` as a read-only fixture root. Tests must
write generated output to a temporary directory, never beside source photographs.
