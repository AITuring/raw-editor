# Third-party notices

## Compact ICC Profiles

- Component: `sRGB-v4.icc` (480 bytes; embedded as Base64 in `color_management.rs`)
- Project: [Compact ICC Profiles](https://github.com/saucecontrol/Compact-ICC-Profiles)
- Upstream file: `profiles/sRGB-v4.icc`
- SHA-256: `c56e1685d888f5edb92fe07f2750f387f8fe8e91b32ff8fb0b56bfbbb9458353`
- License: CC0-1.0 / public-domain dedication
- Local use: embedded in supported export formats as the default sRGB output profile; bytes are unmodified

## moxcms

- Component: `moxcms` 0.8.1
- Project: [moxcms](https://github.com/awxkee/moxcms)
- License: BSD-3-Clause OR Apache-2.0
- Local use: parse untrusted embedded RGB ICC profiles and convert decoded non-RAW pixels to the editor's sRGB input contract
- Integration: used as an unmodified Cargo dependency; no moxcms source is copied into this repository

## libwebp-sys / libwebp

- Component: `libwebp-sys` 0.9.6 and its vendored Google libwebp source
- Projects: [libwebp-sys](https://github.com/NoXF/libwebp-sys), [libwebp](https://github.com/webmproject/libwebp)
- Licenses: MIT (`libwebp-sys` Rust wrapper); BSD-3-Clause (libwebp)
- Local use: desktop lossy WebP imports directly into a YUVA picture and sends encoded chunks to an adjacent temporary file through libwebp's writer callback
- Integration: used as an unmodified Cargo dependency; no upstream source is copied into application source files

Existing bundled libraries, model code and the Lensfun database retain their upstream notices and
licenses. See `NOTICE`, dependency manifests and the source headers distributed with this repository.
