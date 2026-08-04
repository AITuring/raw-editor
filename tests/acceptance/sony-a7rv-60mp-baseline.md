# Sony α7R V 60 MP real-RAW baseline

## Status

Partially established on 2026-08-04 with the user-authorized `DSC05363.ARW` fixture. This result
covers the `lossy-arw` acceptance scenario only. It is a real-camera pipeline-quality and local
performance baseline, not a complete α7R V image-quality verdict.

The source hash, authorization, sensor dimensions and expected RAW properties are authoritative in
[`raw-manifest.json`](raw-manifest.json). The source ARW remains local and Git-ignored.

## Sample and run

- Camera: Sony ILCE-7RM5 (α7R V), firmware 4.00
- Capture: ISO 125, 25 s, f/8, 20 mm, auto white balance
- RAW mode: Sony compressed RAW (lossy), 14-bit RGGB
- Source: 73,781,248 bytes; mosaic 9600 × 6376
- Output: correctly oriented 6336 × 9504 (60.2 MP)
- Run: macOS arm64, release profile, 12 logical CPUs, 20,180,369,408 available memory bytes at start

The ignored real-RAW acceptance test passed the source size and SHA-256 gate, camera identity, mosaic
dimensions, source/decoded bit depths, CFA, black/white levels, positive as-shot white balance, color
matrix availability, node-level transformations, preview/full consistency, output dimensions and ICC
profile checks.

## Node observations

The centered 512 × 512 real-mosaic probe produced distinct finite outputs for every processing node:

| Node                         |    Time | Observed result                            |
| ---------------------------- | ------: | ------------------------------------------ |
| Rescale                      | 0.48 ms | Preserved one-channel CFA mosaic           |
| Quality demosaic             | 3.99 ms | Produced three-channel camera RGB          |
| Neutral-WB color calibration | 3.12 ms | Changed output; mean absolute delta 0.0210 |
| As-shot-WB color calibration | 2.86 ms | Changed output; mean absolute delta 0.0521 |

These probe timings describe a 512-pixel crop and are node diagnostics, not full-image performance.

## Local performance observation

| Operation                                             |        Time |
| ----------------------------------------------------- | ----------: |
| Isolated RAW unpack                                   |    28.42 ms |
| Production full development, including its own decode | 1,767.48 ms |
| CPU default display transform                         |   171.04 ms |
| 1280 × 1920 preview resize                            |    11.75 ms |
| Preview JPEG encode, quality 92                       |    40.67 ms |
| 6336 × 9504 JPEG encode, quality 92                   |   889.88 ms |
| Complete fixture run                                  | 3,064.56 ms |

The preview was 869,097 bytes and the full-resolution JPEG was 16,546,655 bytes. Both carried the
expected 480-byte sRGB v4 ICC profile. Preview and full-resolution sampled channel means differed by
at most 0.00123. These are one local release run's observations; they are not pass/fail thresholds and
should be remeasured after pipeline or dependency changes.

## Scoped image-quality conclusion

- Orientation, crop and output dimensions match the camera metadata.
- The complete image and inspected 100% crops show resolved fine surface detail without an obvious
  checkerboard, channel displacement, zipper edge or large false-color defect.
- White balance and camera-to-RGB processing produce plausible color, and preview/full-resolution
  global channel means remain closely aligned.
- A manual comparison with the full embedded camera JPEG confirms matching geometry and scene
  structure. The editor's CPU-default rendering is visibly darker with deeper shadows and a different
  tone/color rendering. The embedded JPEG is camera-processed and is not an Adobe Camera Raw
  reference.
- The linear developed sample retains values above 1.0, while the current CPU display-derived JPEG
  clamps to the display range. A bright opening is visibly clipped in that baseline output. This
  finding applies to the CPU acceptance path; the GPU tone-mapped application export was not measured
  by this harness.
- Apparent local softness cannot be attributed to demosaic quality from this sample alone because it
  is a 25-second capture without a paired reference rendering or controlled focus target.

Therefore this sample is sufficient to detect major unpacking, demosaic, white-balance,
color-transform, preview and encoder regressions and to anchor local 60 MP timing. It is not sufficient
to claim absolute color accuracy, Adobe Camera Raw parity, highlight-rendering parity, noise quality
across ISO values or universal α7R V detail performance.

## Remaining α7R V coverage

The following scenarios remain deferred: daylight, tungsten, high ISO, underexposure, saturated
colored light, lossless-L ARW and uncompressed ARW. A controlled color/detail target and paired
reference rendering are also required before making absolute color or processor-parity claims.
