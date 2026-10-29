# 源码布局：facade 与子模块

2026-10-28/29 的两轮拆分把仓库里超过 1000 行的 Rust、TypeScript 与 CSS 文件拆成「facade + 同名目录下的子模块」。
本文说明这套布局的约定、怎样验证拆分只移动了代码，以及每个 facade 下的模块地图和仍超过 1000 行的文件。
Focus Stack 的阶段边界与验收证据见 [`focus-stack-architecture.md`](focus-stack-architecture.md)。

## 约定

### Rust

- `foo.rs` 保留为 facade，子模块放在 `foo/<child>.rs`，facade 用 `mod child;` 声明；只在测试构建中编译的子模块用
  `#[cfg(test)] mod child;`。facade 只保留被多个子模块共享的常量与小 helper、`use`、模块声明、兼容重导出、
  已有的 `#[path]` 兄弟模块声明，以及 Tauri 命令（见下）。
- 外部路径保持不变：被移动的条目按原可见性在 facade 重导出，`crate::foo::X` 的写法和可见范围都不变。私有条目移入子
  模块后加 `pub(super)`；原来是 `pub(super)` 的条目改为 `pub(in <facade 的父模块>)`，可见范围相同。整个 crate 内无人
  使用的重导出会被 rustc 报为 unused import，因此删除（条目本身仍在子模块中保持原可见性）。
- 移入子模块的代码里，指向兄弟模块的相对路径（`super::x::…`）改写为绝对路径（如 `crate::panorama_utils::x::…`），
  指向同一个模块。`include_str!`/`include_bytes!` 的相对路径按文件所在目录解析，文件深一层就多一级 `../`，读取的仍是同一文件。
- 通过 `#[path]` 加载的文件（如 `panorama_virtual_tile_tests.rs`、`panorama_structural_oracle.rs`、
  `panorama_utils/final_world_owner_oracle.rs`、`raw_processing_acceptance.rs`）会被 rustc 当作 `mod.rs` 一类的文件：
  普通 `mod child;` 会在同一目录查找 `child.rs`。这类文件的子模块写成 `#[path = "<stem>/<child>.rs"] mod <child>;`，
  文件仍放在 `<stem>/` 目录。crate 根 `lib.rs` 的子模块放在 `src/lib/`，同样用 `#[path = "lib/<child>.rs"]` 声明。
- Tauri 命令：`#[tauri::command]` 会在函数旁生成隐藏宏，`lib.rs` 的 `generate_handler!` 按函数所在的模块路径引用它们，
  `pub use` 不会转发这些宏。所以命令函数一般留在 facade；`file_management` 与 `lib.rs` 的 52 个命令随实现移入子模块，
  `generate_handler!` 中对应条目原位改为完整路径（如 `file_management::listing::list_images_in_dir`）。前端调用的命令名取自
  函数名，不受影响。
- 每个子模块开头用 `//!` 写一句职责说明，下方模块地图的“说明”列即取自这一行。

### 测试

- 内联测试模块原样外置为 `<module>/tests.rs`（或原名 `<name>_tests.rs`），再按主题拆成 `<module>/tests/<topic>.rs`。
  测试函数名和测试总数不变（943 个，其中 40 个 ignored）；拆成主题文件的测试路径多出一段，例如
  `panorama_utils::mosaic::tests::focus_ownership::…`。完整的旧路径 → 新路径对照见
  [`test-path-map.tsv`](test-path-map.tsv)，原模块前缀始终是新路径的前缀。
- `package.json` 中用 `--exact` 调用的 `stack_acceptance_harness` 与 `stack_station_pair_evidence_harness` 留在
  `stack_pipeline/properties.rs` facade，路径不变。真实素材回归按名字过滤 `real_focus_stack_fixture_from_env`（不加
  `--exact`），它现在位于 `panorama_stitching::acceptance_tests::focus_stack_fixture`。
- proptest 的失败样本按测试所在源文件定位：`src-tauri/src/a/b.rs` 对应 `src-tauri/proptest-regressions/a/b.txt`，文件里的每条
  `cc` 种子都会对该源文件内的所有 proptest 回放。移动 proptest 时，种子文件随之移动或复制到每个新文件的对应路径，
  例如 `stack_pipeline/compositor.txt` → `compositor/tests.txt`，`properties.txt` → `properties/<domain>.txt`（21 份相同内容）。

### TypeScript / TSX

- `Name.tsx`（或 `.ts`）保留主组件、主 hook 或模块入口；其余顶层声明（子组件、helper、常量、类型）移到同目录下与文件主干
  同名的目录 `Name/<Part>.tsx|ts`。原文件的导出名不变（`export { X } from './Name/part'`），所有外部 import 路径不变。
- 只移动顶层声明，不改函数体、JSX、hook 调用顺序、className、i18n key 和注释；part 不 import 原文件，不产生循环依赖；
  MasksPanel 与 AIPanel 中同名的子组件各自保留一份。
- 目录名必须与文件主干大小写完全一致。macOS 文件系统不区分大小写，`src/components/panel/Editor.tsx` 旁已有 `editor/`
  目录，所以 `Editor.tsx` 不拆。

### CSS

- `src/styles.css` 先引入 `styles/tokens.css` 与 Tailwind，保留 `@source`（路径相对于该文件），然后按原顺序用相对 `@import`
  引入 `src/styles/app/` 下的 12 个文件。只在顶层规则之间切分；原来很长的一个 `@layer components` 块拆成四个相邻的同名
  `@layer components` 块，Tailwind 构建时会合并它们。

### 源码契约

`tests/*.mjs` 中的源码契约通过 [`tests/support/rust-source.mjs`](../tests/support/rust-source.mjs) 与
[`tests/support/source-module.mjs`](../tests/support/source-module.mjs) 读取源码：Rust 模块读 `foo.rs` 加 `foo/` 下的全部子模块，
TypeScript 读 `Name.tsx` 加 `Name/` 下的 parts，CSS 把相对 `@import` 就地展开。检查因此跟随实现，而不依赖单文件布局。

## 怎样验证“只移动了代码”

每个文件都在独立 worktree 中逐步拆分，每一步都要通过：

- `cargo check --lib --tests` 的警告消息集合（不含位置）与拆分前完全相同；`cargo fmt --check` 与 `git diff --check` 干净；
- 条目级核对：拆分前的每个顶层条目（`use` 除外）在 facade 或子模块中恰好出现一次，归一化空白、可见性与上述路径限定后
  文本相同；外置测试模块与 rustfmt 处理后的原模块体逐字节相同。前端用同样规则按 TypeScript AST 核对每个顶层语句，并核对注释；
- release 全量库测试 `903 passed; 0 failed; 40 ignored`，测试清单按名字与拆分前一一对应；
- 14 个 Node 源码契约全部通过；前端另跑 `typecheck`、`lint`、`build`、`i18n:check`，构建出的 CSS 与拆分前逐字节相同；
- 兰亭与马远 PSD 的 10% 真实素材回归 verdict 为 `identical`（preview 与六个裁片 digest 与基线一致）。

合并后的完整树再执行一遍上述检查。

## 模块地图

下面的表格由 `python3 scripts/source-layout.py --write` 从当前源码生成，模块增删或行数变化后重新运行即可。

<!-- source-layout:begin -->

### Rust：facade 与子模块

#### `ai_processing.rs`（facade 126 行，7 个子模块）

| 子模块               | 行数 | 仅测试 | 说明                                                          |
| -------------------- | ---: | :----: | ------------------------------------------------------------- |
| `denoise.rs`         |  262 |        | Tiled NIND AI denoising.                                      |
| `lama_inpainting.rs` |  154 |        | LaMa inpainting around a mask.                                |
| `mask_parameters.rs` |   78 |        | Parameters of AI sub-masks as stored in the adjustments JSON. |
| `model_files.rs`     |  280 |        | Model and ONNX Runtime files on disk.                         |
| `model_sessions.rs`  |  331 |        | Lazily initialised ONNX sessions held in AiState.             |
| `sam.rs`             |  362 |        | Segment Anything subject masks.                               |
| `segmentation.rs`    |  262 |        | Sky, foreground and depth model passes.                       |

#### `crop_analysis.rs`（facade 56 行，5 个子模块）

| 子模块                | 行数 | 仅测试 | 说明                                    |
| --------------------- | ---: | :----: | --------------------------------------- |
| `axis_consensus.rs`   |  374 |        | Axis consensus fitting for Upright.     |
| `straighten.rs`       |  228 |        | Automatic straighten analysis.          |
| `structural_lines.rs` |  290 |        | Structural line extraction for Upright. |
| `tests.rs`            |  356 |   ✓    | 单元测试（原内联 `mod tests`）          |
| `upright.rs`          |  457 |        | Upright perspective analysis.           |

#### `denoising.rs`（facade 278 行，5 个子模块）

| 子模块              | 行数 | 仅测试 | 说明                                        |
| ------------------- | ---: | :----: | ------------------------------------------- |
| `block_matching.rs` |  153 |        | Block matching for BM3D.                    |
| `bm3d.rs`           |  235 |        | Joint BM3D filtering of the Y/Cb/Cr planes. |
| `pipeline.rs`       |  309 |        | Image-level denoising jobs.                 |
| `tests.rs`          |   99 |   ✓    | 单元测试（原内联 `mod tests`）              |
| `transforms.rs`     |  165 |        | BM3D transforms.                            |

#### `exif_processing.rs`（facade 51 行，6 个子模块）

| 子模块               | 行数 | 仅测试 | 说明                                             |
| -------------------- | ---: | :----: | ------------------------------------------------ |
| `exif_sidecar.rs`    |  208 |        | The EXIF map cached in the primary sidecar.      |
| `exif_values.rs`     |  209 |        | EXIF value conversion and display formatting.    |
| `export_metadata.rs` |  617 |        | Metadata written into exported files.            |
| `metadata_read.rs`   |  607 |        | Reading EXIF and RAW metadata from source files. |
| `sidecar.rs`         |  233 |        | Image sidecar documents on disk.                 |
| `tests.rs`           |  132 |   ✓    | 单元测试（原内联 `mod tests`）                   |

#### `export_processing.rs`（facade 872 行，16 个子模块）

| 子模块                            | 行数 | 仅测试 | 说明                                                                |
| --------------------------------- | ---: | :----: | ------------------------------------------------------------------- |
| `batch_export.rs`                 |  648 |        | The batch export loop and the headless export entry.                |
| `batch_geometry.rs`               |  201 |        | Batch geometry correction: result types, lens data and adjustments. |
| `batch_geometry_preview.rs`       |  205 |        | Batch geometry review previews.                                     |
| `content_orientation.rs`          |  156 |        | Content orientation analysis for batch geometry review.             |
| `encode_bytes.rs`                 |  294 |        | In-memory export encoders.                                          |
| `export_render.rs`                |  477 |        | Rendering one export source.                                        |
| `file_output.rs`                  |  142 |        | Writing encoded exports to disk.                                    |
| `masks_and_lut.rs`                |  241 |        | Mask and LUT exports that accompany an exported image.              |
| `settings.rs`                     |  147 |        | Export settings and their defaults.                                 |
| `streaming_encoders.rs`           |  380 |        | Row-streaming export encoders.                                      |
| `tasks.rs`                        |  184 |        | Export task registration, cancellation and progress events.         |
| `tests.rs（另见下方同名 facade）` |  122 |   ✓    | 单元测试（原内联 `mod tests`）                                      |
| `tiff_streaming.rs`               |  366 |        | Row-streaming TIFF and BigTIFF encoders.                            |
| `tiff_writers.rs`                 |  374 |        | Whole-image TIFF and BigTIFF writers with metadata.                 |
| `watermark.rs`                    |  323 |        | Watermarks: library import, preparation and compositing.            |
| `webp_output.rs`                  |  503 |        | Bounded-memory WebP output.                                         |

#### `export_processing/tests.rs`（facade 122 行，仅测试，7 个子模块）

| 子模块                     | 行数 | 仅测试 | 说明                                        |
| -------------------------- | ---: | :----: | ------------------------------------------- |
| `batch_jobs.rs`            |  108 |   ✓    | Batch export jobs.                          |
| `bounded_webp.rs`          |  135 |   ✓    | Bounded-memory WebP output.                 |
| `encoded_formats.rs`       |  158 |   ✓    | In-memory and BigTIFF encoders.             |
| `large_image_harnesses.rs` |  256 |   ✓    | Ignored 60 MP export benchmarks.            |
| `streaming_metadata.rs`    |  256 |   ✓    | EXIF and TIFF metadata in streamed exports. |
| `streaming_output.rs`      |  341 |   ✓    | Row-streaming export.                       |
| `watermarking.rs`          |  107 |   ✓    | Watermarks in exports.                      |

#### `file_management.rs`（facade 271 行，11 个子模块）

| 子模块                | 行数 | 仅测试 | 说明                                                           |
| --------------------- | ---: | :----: | -------------------------------------------------------------- |
| `albums.rs`           |  293 |        | Album commands and album path bookkeeping.                     |
| `file_operations.rs`  |  693 |        | File and folder operation commands.                            |
| `folder_tree.rs`      |  281 |        | Folder tree commands.                                          |
| `importing.rs`        |  232 |        | The import command and its file-name templates.                |
| `listing.rs`          |  444 |        | Folder listing commands.                                       |
| `metadata.rs`         |  136 |        | Image metadata resolution and the background metadata workers. |
| `presets.rs`          |  208 |        | Preset storage and the preset import and export commands.      |
| `sidecar_edits.rs`    |  627 |        | Commands that edit image sidecars.                             |
| `thumbnail_queue.rs`  |  396 |        | Thumbnail cache, queue and worker threads.                     |
| `thumbnail_render.rs` |  423 |        | Thumbnail rendering.                                           |
| `xmp_sync.rs`         |  236 |        | XMP sidecar synchronisation.                                   |

#### `gpu_processing.rs`（facade 150 行，5 个子模块）

| 子模块                                | 行数 | 仅测试 | 说明                                                              |
| ------------------------------------- | ---: | :----: | ----------------------------------------------------------------- |
| `context.rs`                          |  396 |        | GPU context initialisation and the native wgpu display.           |
| `processor.rs（另见下方同名 facade）` |   77 |        | The GPU processor: pipelines, textures and the tiled render pass. |
| `render.rs`                           |  555 |        | Render entry points over the shared GPU processor.                |
| `shader_tests.rs`                     |  304 |   ✓    | 单元测试（原内联 `mod shader_tests`）                             |
| `texture_io.rs`                       |  291 |        | Texture uploads and readbacks.                                    |

#### `gpu_processing/processor.rs`（facade 77 行，2 个子模块）

| 子模块         | 行数 | 仅测试 | 说明                                                         |
| -------------- | ---: | :----: | ------------------------------------------------------------ |
| `pipelines.rs` |  573 |        | `GpuProcessor::new`: compute pipelines and working textures. |
| `tiled_run.rs` |  675 |        | `GpuProcessor::run`: the tiled render pass.                  |

#### `image_loader.rs`（facade 195 行，3 个子模块）

| 子模块                   | 行数 | 仅测试 | 说明                                         |
| ------------------------ | ---: | :----: | -------------------------------------------- |
| `color_profile_tests.rs` |  138 |   ✓    | 单元测试（原内联 `mod color_profile_tests`） |
| `patch_compositing.rs`   |  423 |        | AI patch compositing on the base image.      |
| `source_decode.rs`       |  434 |        | Decoding source bytes into the base image.   |

#### `image_processing.rs`（facade 168 行，18 个子模块）

| 子模块                        | 行数 | 仅测试 | 说明                                                          |
| ----------------------------- | ---: | :----: | ------------------------------------------------------------- |
| `adjustment_parsing.rs`       |  683 |        | Adjustments JSON to GPU uniforms.                             |
| `adjustments.rs`              |  223 |        | GPU adjustment uniforms.                                      |
| `auto_adjust.rs`              |  398 |        | Automatic adjustment analysis.                                |
| `auto_white_balance_tests.rs` |   41 |   ✓    | 单元测试（原内联 `mod auto_white_balance_tests`）             |
| `color_transfer_tests.rs`     |   26 |   ✓    | 单元测试（原内联 `mod color_transfer_tests`）                 |
| `color_transforms.rs`         |  323 |        | CPU colour transforms and the AgX tone mapper.                |
| `downscale.rs`                |  167 |        | Float-preserving image downscaling.                           |
| `geometry_ops.rs`             |  429 |        | Image-level geometry operations.                              |
| `geometry_params.rs`          |  236 |        | Geometry parameters of an edit.                               |
| `geometry_projection.rs`      |  144 |        | Geometry transform matrices and projection mapping.           |
| `geometry_sampling.rs`        |  317 |        | Source sampling for the geometry warp.                        |
| `geometry_transform_tests.rs` |  906 |   ✓    | 单元测试（原内联 `mod geometry_transform_tests`）             |
| `geometry_warp.rs`            |  484 |        | The forward geometry warp.                                    |
| `geometry_warp_rows.rs`       |  496 |        | Row/band geometry output for the full-resolution export path. |
| `raw_enhance.rs`              |  241 |        | RAW artifact removal and gentle detail enhancement.           |
| `raw_enhancement_tests.rs`    |   53 |   ✓    | 单元测试（原内联 `mod raw_enhancement_tests`）                |
| `scopes.rs`                   |  458 |        | Histogram and waveform scopes.                                |
| `sidecar.rs`                  |  113 |        | Sidecar metadata and the edited-state check.                  |

#### `image_stack.rs`（facade 373 行，7 个子模块）

| 子模块                 | 行数 | 仅测试 | 说明                                                            |
| ---------------------- | ---: | :----: | --------------------------------------------------------------- |
| `output_encoding.rs`   |  278 |        | Encoding the canonical image-stack result.                      |
| `output_format.rs`     |  144 |        | Image-stack output formats and the fidelity notices they imply. |
| `output_validation.rs` |  198 |        | Read-back validation of written image-stack files.              |
| `output_writer.rs`     |  236 |        | Publishing image-stack outputs.                                 |
| `preview_files.rs`     |  183 |        | UI preview files for a finished image stack.                    |
| `test_output.rs`       |   87 |   ✓    | Test-only image-stack output helpers.                           |
| `tests.rs`             |  875 |   ✓    | 单元测试（原内联 `mod tests`）                                  |

#### `lens_blur.rs`（facade 83 行，3 个子模块）

| 子模块         | 行数 | 仅测试 | 说明                                                           |
| -------------- | ---: | :----: | -------------------------------------------------------------- |
| `bokeh.rs`     |  229 |        | Aperture-shaped bokeh kernels and per-layer blur.              |
| `coc_field.rs` |  290 |        | Signed circle-of-confusion field from the depth map.           |
| `render.rs`    |  496 |        | Lens-blur entry point and the layered depth-of-field renderer. |

#### `lens_correction.rs`（facade 146 行，5 个子模块）

| 子模块                 | 行数 | 仅测试 | 说明                                                   |
| ---------------------- | ---: | :----: | ------------------------------------------------------ |
| `lens_profile.rs`      |  390 |        | Lensfun lens records and their calibration lookup.     |
| `lens_search.rs`       |  141 |        | Finding a Lensfun profile for camera metadata.         |
| `lensfun_db.rs`        |  228 |        | Lensfun database records and loading.                  |
| `stack_calibration.rs` |  348 |        | Calibrated lens geometry for focus-stack registration. |
| `tests.rs`             |  220 |   ✓    | 单元测试（原内联 `mod tests`）                         |

#### `lib.rs`（facade 748 行，9 个子模块）

| 子模块                               | 行数 | 仅测试 | 说明                                                           |
| ------------------------------------ | ---: | :----: | -------------------------------------------------------------- |
| `geometry_previews.rs`               |  571 |        | Crop and geometry previews and the crop analysis commands.     |
| `logging.rs`                         |  109 |        | Application logging setup and the logging commands.            |
| `merged_outputs.rs`                  |  152 |        | HDR merge and save, and collage save commands.                 |
| `native_preview_display_tests.rs`    |   20 |   ✓    | 单元测试（原内联 `mod native_preview_display_tests`）          |
| `preview_cache.rs`                   |  163 |        | Transformed preview images and the full-resolution warp cache. |
| `preview_commands.rs`                |  325 |        | Editor preview commands.                                       |
| `preview_worker.rs`                  |  420 |        | The editor preview worker and the analytics worker.            |
| `transformed_image_sharing_tests.rs` |   26 |   ✓    | 单元测试（原内联 `mod transformed_image_sharing_tests`）       |
| `window_state.rs`                    |  168 |        | Saved window state checks and the frontend-ready command.      |

#### `mask_generation.rs`（facade 346 行，7 个子模块）

| 子模块                            | 行数 | 仅测试 | 说明                                                       |
| --------------------------------- | ---: | :----: | ---------------------------------------------------------- |
| `ai_masks.rs`                     |  526 |        | AI sub-masks from stored mask bitmaps.                     |
| `composition.rs`                  |  591 |        | Sub-mask generation and mask composition.                  |
| `grow_feather.rs`                 |  157 |        | Grow and feather post-processing for sub-masks.            |
| `parameters.rs`                   |  144 |        | Sub-mask parameter types parsed from the adjustments JSON. |
| `range_masks.rs`                  |  287 |        | Colour and luminance range sub-masks.                      |
| `shape_masks.rs`                  |  461 |        | Radial, linear, brush and flow sub-masks.                  |
| `tests.rs（另见下方同名 facade）` |   22 |   ✓    | 单元测试（原内联 `mod tests`）                             |

#### `mask_generation/tests.rs`（facade 22 行，仅测试，4 个子模块）

| 子模块                     | 行数 | 仅测试 | 说明                                                 |
| -------------------------- | ---: | :----: | ---------------------------------------------------- |
| `composition_and_cache.rs` |  120 |   ✓    | Mask composition and cache ownership.                |
| `large_mask_harnesses.rs`  |  637 |   ✓    | Ignored 60 MP mask benchmarks.                       |
| `scratch_bounds.rs`        |   85 |   ✓    | Scratch-memory bounds for 60 MP masks.               |
| `tiled_parity.rs`          |  497 |   ✓    | Tiled mask generation against full-frame generation. |

#### `panorama_stitching.rs`（facade 936 行，43 个子模块）

| 子模块                                       |  行数 | 仅测试 | 说明                                                                    |
| -------------------------------------------- | ----: | :----: | ----------------------------------------------------------------------- |
| `acceptance_tests.rs（另见下方同名 facade）` |    14 |   ✓    | 单元测试（原内联 `mod acceptance_tests`）                               |
| `alignment_cache.rs`                         |   236 |   ✓    | Test-only alignment cache writer.                                       |
| `alignment_tests.rs（另见下方同名 facade）`  |   167 |   ✓    | 单元测试（原内联 `mod alignment_tests`）                                |
| `capture_sequence_bridges.rs`                |   696 |        | Capture-sequence and overview-gap bridges in the focus match graph.     |
| `determinism_test_access.rs`                 |   116 |   ✓    | 测试访问入口（原内联 `mod determinism_test_access`）                    |
| `flat_artwork.rs`                            |   428 |        | Flat-artwork render order, tree-pose refinement and the blur filter.    |
| `focus_auto_order.rs`                        |   516 |        | Automatic focus-stack ordering.                                         |
| `focus_capture_geometry.rs`                  |   660 |        | Capture geometry checks and compositor order for focus stacks.          |
| `focus_capture_spine.rs`                     |   410 |        | Focus capture stations and the spine of the focus match graph.          |
| `focus_component_recovery.rs`                |   555 |        | Recovery of disconnected focus match components.                        |
| `focus_global_observations.rs`               |   548 |        | Pose parametrisation and observations of the global focus solve.        |
| `focus_global_solver.rs`                     |   454 |        | Global focus-stack homography optimisation.                             |
| `focus_layer_warp.rs`                        |   522 |        | Focus layer warps, band constraints and foreground edge alignment.      |
| `focus_stack_order.rs`                       |   393 |        | The focus-stack stitching order and its report.                         |
| `grouping_test_access.rs`                    |   136 |   ✓    | 测试访问入口（原内联 `mod grouping_test_access`）                       |
| `horizontal_edge_bands.rs`                   |   631 |        | Horizontal edge lines and the focus warp bands derived from them.       |
| `options.rs`                                 |    69 |        | Stitching modes and the outcome of a stitch run.                        |
| `overlap_quality.rs`                         |   560 |        | Overlap quality and spatial support of a pair transform.                |
| `pair_matching.rs`                           |   590 |        | Pairwise image matching.                                                |
| `patch_refine.rs`                            |   759 |        | Patch-level match refinement and dense point sampling.                  |
| `run_control.rs`                             |   277 |        | Run-scoped resource guards, progress events and focus-stack cache keys. |
| `source_prep.rs`                             |   585 |        | Input preflight and source preparation.                                 |
| `source_raw_direct.rs`                       |   664 |        | Direct photometric refinement for Source_RAW layer pairs.               |
| `source_raw_direct_basin.rs`                 |   466 |        | Basin selection of the direct Source_RAW refinement.                    |
| `source_raw_direct_pairs.rs`                 |   416 |        | Direct Source_RAW pair observations and their consensus.                |
| `source_raw_layer_diagnostics.rs`            |   782 |        | Source_RAW layer-pair diagnostics.                                      |
| `station_relation_test_access.rs`            |   424 |   ✓    | 测试访问入口（原内联 `mod station_relation_test_access`）               |
| `stitch_images.rs`                           | 3,926 |        | The stitching pipeline body, `stitch_images_with_options`.              |
| `stitching_order.rs`                         |   342 |        | Panorama stitching order.                                               |
| `transform_math.rs`                          |   577 |        | Transform estimation and robust-fit primitives.                         |
| `transform_selection.rs`                     |   507 |        | Model selection for panorama and focus-stack pair transforms.           |
| `vertical_edge_bands.rs`                     |   626 |        | Vertical edge lines and the focus warp bands derived from them.         |
| `virtual_tile_direct_match.rs`               |   691 |        | Bounded direct probes between Virtual_Tile stations.                    |
| `virtual_tile_geometry_repair.rs`            |   489 |        | Rendering-prior repair and joint geometry refinement of Virtual_Tiles.  |
| `virtual_tile_global_search.rs`              |   491 |        | Prior-free and source-match proposals for Virtual_Tile relations.       |
| `virtual_tile_guided_match.rs`               |   604 |        | Descriptor-guided Virtual_Tile station matching.                        |
| `virtual_tile_hypothesis.rs`                 |   419 |        | Selection among Virtual_Tile relation hypotheses.                       |
| `virtual_tile_planes.rs`                     |   656 |        | Virtual_Tile match planes and pyramids.                                 |
| `virtual_tile_polish.rs`                     |   613 |        | Finest-octave polish of a Virtual_Tile relation.                        |
| `virtual_tile_pyramid_match.rs`              |   489 |        | Coarse-to-fine Virtual_Tile station matching.                           |
| `virtual_tile_relations.rs`                  |   460 |        | Virtual_Tile relation measurement and candidate records.                |
| `virtual_tile_residual.rs`                   |   647 |        | Measured residual models of a Virtual_Tile station relation.            |
| `virtual_tile_structure.rs`                  |   311 |        | Structural scores and coverage geometry of Virtual_Tile overlaps.       |

#### `panorama_stitching/acceptance_tests.rs`（facade 14 行，仅测试，6 个子模块）

| 子模块                    | 行数 | 仅测试 | 说明                                                                                       |
| ------------------------- | ---: | :----: | ------------------------------------------------------------------------------------------ |
| `entry_point_fixtures.rs` |  181 |   ✓    | Whole entry-point runs on small fixtures: an undecodable source, the synthetic 200-image   |
| `focus_stack_fixture.rs`  |  701 |   ✓    | The real focus-stack fixture harness run by tests/focus-stack-regression.mjs, its layered  |
| `native_capture.rs`       |  112 |   ✓    | Diagnostic captures that production code calls in test builds (quality_gate_runner and     |
| `ordered_panorama.rs`     |  742 |   ✓    | Ordered panorama fixture diagnostics: pairwise matching and the full render of an external |
| `pair_diagnostics.rs`     |  707 |   ✓    | Pair diagnostics on real inputs: saved Virtual_Tile planes, focus pairs from               |
| `saved_native_replays.rs` |  623 |   ✓    | Replays of saved native diagnostics named by environment variables: contour views, tone    |

#### `panorama_stitching/alignment_tests.rs`（facade 167 行，仅测试，14 个子模块）

| 子模块                           | 行数 | 仅测试 | 说明                                                                                         |
| -------------------------------- | ---: | :----: | -------------------------------------------------------------------------------------------- |
| `capture_grouping.rs`            |  480 |   ✓    | Capture_Station grouping: the four pieces of link evidence, sequence and motion boundaries,  |
| `closure_optimizer.rs`           |  404 |   ✓    | Station closure: projective tree closure and its seed, relation sample points, the robust    |
| `edge_bands.rs`                  |  174 |   ✓    | Long horizontal and vertical edge lines behind the focus warp bands: consensus across        |
| `focus_scan_order.rs`            |  433 |   ✓    | Focus-stack scan geometry and stitching order: geometry guards, shifted-structure and        |
| `group_pose_solver.rs`           |  366 |   ✓    | Capture-group pose solving: disconnected station graphs, coarse consensus and its conflicts, |
| `match_pairs.rs`                 |  205 |   ✓    | Pair matching contracts: candidate pairs for large and medium stacks, natural filename       |
| `relation_acceptance.rs`         |  268 |   ✓    | Station_Relation acceptance: the literal gate boundaries and one identifier per failed       |
| `render_ordering.rs`             |  271 |   ✓    | Render and stitching order: the flat-artwork overlap walk, compositor order over the station |
| `source_preparation.rs`          |  158 |   ✓    | Source preparation and run budgets: RAW display gain and exposure normalisation,             |
| `transform_models.rs`            |  404 |   ✓    | Transform estimation and model selection: native and robust affine fits, panorama and        |
| `virtual_tile_matching.rs`       |  779 |   ✓    | Virtual_Tile station matching: descriptor-guided and direct grids, polished and              |
| `virtual_tile_measurements.rs`   |  235 |   ✓    | Virtual_Tile coverage and measurements: covered features, exposure compensation and gradient |
| `virtual_tile_residual_model.rs` |  180 |   ✓    | Measured residual models of Virtual_Tile relations: translation first, bounded affine        |
| `virtual_tile_solver_flow.rs`    |  276 |   ✓    | The Virtual_Tile station solver end to end: key-direction relations in either content        |

#### `panorama_structural_oracle.rs`（facade 166 行，仅测试，4 个子模块）

| 子模块                     | 行数 | 仅测试 | 说明                                                                                       |
| -------------------------- | ---: | :----: | ------------------------------------------------------------------------------------------ |
| `direction_solver.rs`      |  407 |   ✓    | One solve direction: mutual descriptor matches, geometric-hash similarity proposals, model |
| `edge_chains.rs`           |  179 |   ✓    | Coverage-safe edge planes and the edge chains traced through them.                         |
| `structural_primitives.rs` |  323 |   ✓    | Structural primitives (line endpoints, line intersections and high-curvature points) and   |
| `tests.rs`                 |  249 |   ✓    | Independent synthetic tests of the structural oracle: auditable primitives, uncovered      |

#### `panorama_utils/final_world_owner_oracle.rs`（facade 130 行，仅测试，6 个子模块）

| 子模块             | 行数 | 仅测试 | 说明                                                        |
| ------------------ | ---: | :----: | ----------------------------------------------------------- |
| `exact_regions.rs` |  300 |   ✓    | Exact-cost region proposals and the reduced exact solve.    |
| `exact_runs.rs`    |  280 |   ✓    | Exact per-pixel owner evidence aggregated into sparse runs. |
| `region_solver.rs` |  576 |   ✓    | Coherent-region options and the reduced coarse-grid solve.  |
| `snapshots.rs`     |   73 |   ✓    | Per-station candidate snapshots of the current run.         |
| `tests.rs`         |  487 |   ✓    | 单元测试（原内联 `mod tests`）                              |
| `world_cells.rs`   |  178 |   ✓    | Final-world cells composed from station candidates.         |

#### `panorama_utils/mosaic.rs`（facade 277 行，15 个子模块）

| 子模块                            | 行数 | 仅测试 | 说明                                                                       |
| --------------------------------- | ---: | :----: | -------------------------------------------------------------------------- |
| `focus_measure.rs`                |  342 |        | Focus measures shared by the mosaic and the Focus_Fuser.                   |
| `in_memory_mosaic.rs`             |  466 |        | In-memory detail-preserving focus mosaic.                                  |
| `intra_station_test_access.rs`    |  232 |   ✓    | 测试访问入口（原内联 `mod intra_station_test_access`）                     |
| `layer_refine.rs`                 |  234 |        | Analysis-scale residual refinement of one source layer.                    |
| `layer_tone.rs`                   |  313 |        | Per-layer tone fields.                                                     |
| `native_control_points.rs`        |  553 |        | Native control-point measurement of the Intra_Station_Registrar.           |
| `native_refine.rs`                |  624 |        | Native-resolution residual refinement passes (需求 2.3–2.9).               |
| `sampling.rs`                     |  161 |        | Layer sampling primitives shared by both focus mosaic compositors.         |
| `source_ownership.rs`             |  545 |        | Source ownership decisions.                                                |
| `station_registration.rs`         |  151 |        | Intra_Station_Registrar entry for the default station path (需求 2.2–2.9). |
| `streaming_harmonization.rs`      |  551 |        | Overlap-based exposure harmonization of the streaming focus mosaic.        |
| `streaming_illumination.rs`       |  495 |        | Low-frequency illumination correction of a materialised mosaic.            |
| `streaming_mosaic.rs`             |  458 |        | Streaming (tiled) detail-preserving focus mosaic.                          |
| `streaming_store.rs`              |  593 |        | Tiled on-disk canvas of the streaming focus mosaic and its tile writes.    |
| `tests.rs（另见下方同名 facade）` |   27 |   ✓    | 单元测试（原内联 `mod tests`）                                             |

#### `panorama_utils/mosaic/tests.rs`（facade 27 行，仅测试，4 个子模块）

| 子模块                  | 行数 | 仅测试 | 说明                                                                          |
| ----------------------- | ---: | :----: | ----------------------------------------------------------------------------- |
| `focus_ownership.rs`    |  574 |   ✓    | Source ownership tests: sharpness and sensor noise, disagreement, group       |
| `native_refinement.rs`  |  337 |   ✓    | Native-resolution sampling, control point and intra-station refinement tests. |
| `streaming_tiles.rs`    |  160 |   ✓    | Streaming tile store, coverage bounds and materialisation tests.              |
| `tone_harmonization.rs` |  366 |   ✓    | Low-frequency exposure harmonization, source-owner correction and consensus   |

#### `panorama_utils/processing.rs`（facade 98 行，3 个子模块）

| 子模块          | 行数 | 仅测试 | 说明                                                  |
| --------------- | ---: | :----: | ----------------------------------------------------- |
| `features.rs`   |  737 |        | FAST/BRIEF feature detection and descriptor matching. |
| `homography.rs` |  348 |        | Robust homography estimation.                         |
| `tests.rs`      |  180 |   ✓    | 单元测试（原内联 `mod tests`）                        |

#### `panorama_utils/scan_mosaic.rs`（facade 264 行，7 个子模块）

| 子模块                 | 行数 | 仅测试 | 说明                                               |
| ---------------------- | ---: | :----: | -------------------------------------------------- |
| `compositor.rs`        |  440 |        | Planar scan mosaic entry point.                    |
| `detail_render.rs`     |  235 |        | Owner detail rendering for the planar scan mosaic. |
| `homography_refine.rs` |  497 |        | Registration refinement of the planar tile poses.  |
| `low_frequency.rs`     |  504 |        | Joint photometry and the low-frequency band.       |
| `mesh_correction.rs`   |  454 |        | Smooth per-tile registration correction.           |
| `ownership.rs`         |  288 |        | Ownership planning for the planar scan mosaic.     |
| `tile_analysis.rs`     |  351 |        | Per-tile analysis for the planar scan mosaic.      |

#### `panorama_utils/stack_pipeline/compositor.rs`（facade 896 行，1 个子模块）

| 子模块     | 行数 | 仅测试 | 说明                           |
| ---------- | ---: | :----: | ------------------------------ |
| `tests.rs` |  724 |   ✓    | 单元测试（原内联 `mod tests`） |

#### `panorama_utils/stack_pipeline/degradation.rs`（facade 865 行，1 个子模块）

| 子模块     | 行数 | 仅测试 | 说明                           |
| ---------- | ---: | :----: | ------------------------------ |
| `tests.rs` |  510 |   ✓    | 单元测试（原内联 `mod tests`） |

#### `panorama_utils/stack_pipeline/focus_fuser.rs`（facade 197 行，7 个子模块）

| 子模块                            | 行数 | 仅测试 | 说明                                                                        |
| --------------------------------- | ---: | :----: | --------------------------------------------------------------------------- |
| `cell_metrics.rs`                 |  315 |        | Ownership-cell metrics and grid geometry (需求 3.1, 3.2, 3.3, 3.9).         |
| `confidence.rs`                   |  244 |        | Sharpness_Confidence and low sharpness regions (需求 3.8, 3.9).             |
| `content_regions.rs`              |  319 |   ✓    | Test-only content-region ownership diagnostics.                             |
| `decisive_cells.rs`               |  415 |        | Decisive-cell rules of the station fusion.                                  |
| `ownership_solver.rs`             |  863 |        | Candidate order and the multi-label ownership solver (需求 3.4, 3.5, 3.10). |
| `station_fusion.rs`               |  504 |        | The station fusion driver (需求 3.4, 3.5, 3.6, 3.8, 3.9, 3.10).             |
| `tests.rs（另见下方同名 facade）` |    6 |   ✓    | 单元测试（原内联 `mod tests`）                                              |

#### `panorama_utils/stack_pipeline/focus_fuser/tests.rs`（facade 6 行，仅测试，4 个子模块）

| 子模块                 | 行数 | 仅测试 | 说明                                                                      |
| ---------------------- | ---: | :----: | ------------------------------------------------------------------------- |
| `graph_cut.rs`         |  420 |   ✓    | The multi-label ownership graph cut (需求 3.4, 3.5, 3.10).                |
| `region_decisions.rs`  |  338 |   ✓    | Region level ownership decisions: content-region candidates and their     |
| `sharpness_cells.rs`   |  174 |   ✓    | Sharpness_Score and the ownership cell metrics (需求 3.1, 3.2, 3.3, 3.9). |
| `station_ownership.rs` |  346 |   ✓    | `StationFusion`, the per-station fold the focus renderer drives.          |

#### `panorama_utils/stack_pipeline/intra_station.rs`（facade 917 行，1 个子模块）

| 子模块     | 行数 | 仅测试 | 说明                           |
| ---------- | ---: | :----: | ------------------------------ |
| `tests.rs` |  389 |   ✓    | 单元测试（原内联 `mod tests`） |

#### `panorama_utils/stack_pipeline/joint_warp.rs`（facade 739 行，2 个子模块）

| 子模块             | 行数 | 仅测试 | 说明                                                        |
| ------------------ | ---: | :----: | ----------------------------------------------------------- |
| `decomposition.rs` |  483 |        | Projective/local decomposition of the joint station fields. |
| `tests.rs`         |  474 |   ✓    | 单元测试（原内联 `mod tests`）                              |

#### `panorama_utils/stack_pipeline/properties.rs`（facade 598 行，仅测试，22 个子模块）

| 子模块                          | 行数 | 仅测试 | 说明                                                                           |
| ------------------------------- | ---: | :----: | ------------------------------------------------------------------------------ |
| `acceptance_inputs.rs`          |  400 |   ✓    | Acceptance harness inputs: directories, report selection and source snapshots. |
| `acceptance_verdict.rs`         |  532 |   ✓    | Acceptance_Harness verdict (Property 94).                                      |
| `capture_topology.rs`           |  690 |   ✓    | Capture_Topology_Model and Station_Relation properties (25-29).                |
| `closure.rs`                    |  768 |   ✓    | Closure_Optimizer properties (35-40, 77).                                      |
| `graph_cut.rs`                  |  329 |   ✓    | Property 12: the multi-label ownership graph cut.                              |
| `grouping_invariance.rs`        |  509 |   ✓    | Property 2: station membership and the station level topology are invariant    |
| `intra_station_registration.rs` |  757 |   ✓    | Intra_Station_Registrar properties (6, 7, 8).                                  |
| `ownership_cells.rs`            |  607 |   ✓    | Focus_Fuser ownership cell properties (9, 10, 11).                             |
| `quality_gate_checks.rs`        |  870 |   ✓    | Quality_Gate properties (61-74).                                               |
| `reproducibility.rs`            |  674 |   ✓    | Property 90: the reachable determinism-critical subset of a run is byte for    |
| `run_outcome.rs`                |  823 |   ✓    | Run outcome and resource properties.                                           |
| `single_frame.rs`               |  813 |   ✓    | Single-frame station properties (16, 75).                                      |
| `station_anchor.rs`             |  311 |   ✓    | Property 5: the anchor frame is the sharpest member and maps to itself.        |
| `station_failures.rs`           |  660 |   ✓    | Station level degradation properties (76, 78).                                 |
| `station_membership.rs`         |  657 |   ✓    | Station_Grouper membership properties (1, 4).                                  |
| `station_ownership.rs`          |  762 |   ✓    | Station ownership grid properties (14, 15).                                    |
| `station_pose_graph.rs`         |  635 |   ✓    | Station_Pose_Solver properties (30-34).                                        |
| `station_splits.rs`             |  308 |   ✓    | Property 3: accumulated-motion split positions are uniquely determined.        |
| `tile_cache.rs`                 |  664 |   ✓    | Virtual_Tile_Store disk cache properties (19-22).                              |
| `tile_compositor.rs`            |  499 |   ✓    | Tile_Compositor properties: ownership contract, source samples and seams.      |
| `tile_structure.rs`             |  609 |   ✓    | Virtual_Tile structure properties (13, 18, 23).                                |
| `warp_tone.rs`                  |  726 |   ✓    | Residual_Warp and Tone_Harmonizer properties (41-52).                          |

#### `panorama_utils/stack_pipeline/quality_gate.rs`（facade 113 行，5 个子模块）

| 子模块                   | 行数 | 仅测试 | 说明                                                                  |
| ------------------------ | ---: | :----: | --------------------------------------------------------------------- |
| `boundary_edge_tests.rs` |  125 |   ✓    | 单元测试（原内联 `mod boundary_edge_tests`）                          |
| `boundary_strokes.rs`    |  462 |        | Boundary stroke alignment across Owner_Region boundaries (task 15.6). |
| `gradient_tests.rs`      |   32 |   ✓    | 单元测试（原内联 `mod gradient_tests`）                               |
| `pixel_metrics.rs`       |  664 |        | Pure Quality_Gate pixel measurements on one ROI (tasks 15.3 to 15.5). |
| `roi_geometry.rs`        |  502 |        | Deterministic Quality_Gate ROI geometry (tasks 15.1 and 15.2).        |

#### `panorama_utils/stack_pipeline/quality_gate_runner.rs`（facade 427 行，9 个子模块）

| 子模块                            | 行数 | 仅测试 | 说明                                                                   |
| --------------------------------- | ---: | :----: | ---------------------------------------------------------------------- |
| `boundary_pairs.rs`               |  660 |        | Boundary stroke pairing and large-boundary measurement.                |
| `boundary_repair.rs`              |  879 |        | Repair of failed boundary samples.                                     |
| `coverage.rs`                     |  343 |        | Sparse ROI selection and confidence / effective-pixel coverage.        |
| `criteria.rs`                     |  527 |        | Quality_Gate criteria and verdict rules.                               |
| `research_diagnostics.rs`         |  285 |   ✓    | Test-only gradient-source and effective-coverage research diagnostics. |
| `run_gate.rs`                     |  631 |        | The Quality_Gate run over the final output.                            |
| `source_sampling.rs`              |  508 |        | Source_RAW reference sampling for Quality_Gate ROIs.                   |
| `tests.rs（另见下方同名 facade）` |    8 |   ✓    | 单元测试（原内联 `mod tests`）                                         |
| `tone_owner_oracle.rs`            |  349 |   ✓    | Test-only evidence helpers of the tone owner-fidelity oracle.          |

#### `panorama_utils/stack_pipeline/quality_gate_runner/tests.rs`（facade 8 行，仅测试，5 个子模块）

| 子模块                   | 行数 | 仅测试 | 说明                                                                         |
| ------------------------ | ---: | :----: | ---------------------------------------------------------------------------- |
| `boundary_alignment.rs`  |  249 |   ✓    | Boundary stroke alignment across accepted cross-station seams: the synthetic |
| `coverage_evidence.rs`   |   98 |   ✓    | Confidence and texture evidence over covered pixels, and the sparse ROI      |
| `gate_outcomes.rs`       |  344 |   ✓    | Criterion boundaries and whole-gate outcomes: owner shortfall quantisation,  |
| `reference_sampling.rs`  |  305 |   ✓    | Source_RAW reference ROIs: provenance, the world-to-source mapping and the   |
| `tone_owner_fidelity.rs` |  150 |   ✓    | The tone owner fidelity oracle: owner distribution feasibility, the joint    |

#### `panorama_utils/stack_pipeline/residual_warp.rs`（facade 446 行，2 个子模块）

| 子模块           | 行数 | 仅测试 | 说明                                              |
| ---------------- | ---: | :----: | ------------------------------------------------- |
| `tests.rs`       |  156 |   ✓    | 单元测试（原内联 `mod tests`）                    |
| `warp_region.rs` |  712 |        | One overlap's bounded residual field: WarpRegion. |

#### `panorama_utils/stack_pipeline/test_support.rs`（facade 78 行，仅测试，5 个子模块）

| 子模块               | 行数 | 仅测试 | 说明                                                                 |
| -------------------- | ---: | :----: | -------------------------------------------------------------------- |
| `artwork.rs`         |  139 |   ✓    | The synthetic artwork plane every scan generator crops from.         |
| `capture_station.rs` |  501 |   ✓    | Fixtures inside one Capture_Station.                                 |
| `scan_grid.rs`       |  797 |   ✓    | Synthetic serpentine scans and focus brackets over an artwork plane. |
| `stack_report.rs`    |  315 |   ✓    | Stack_Report generators.                                             |
| `virtual_tiles.rs`   |  426 |   ✓    | Coverage_Mask shapes and Virtual_Tile generators.                    |

#### `panorama_utils/stack_pipeline/tone.rs`（facade 309 行，8 个子模块）

| 子模块                | 行数 | 仅测试 | 说明                                                                     |
| --------------------- | ---: | :----: | ------------------------------------------------------------------------ |
| `color_metrics.rs`    |  117 |        | Colour measurements shared by the Tone_Harmonizer and the Quality_Gate.  |
| `harmonize.rs`        |  661 |        | Tone harmonisation after ownership: the public entry points.             |
| `owner_oracle.rs`     |  301 |   ✓    | Test-only Source_RAW owner-fidelity oracle state of the Tone_Harmonizer. |
| `pair_evidence.rs`    |  524 |        | Pairwise tone evidence between neighbouring tone tiles.                  |
| `test_diagnostics.rs` |  250 |   ✓    | Test-only tone diagnostics writer.                                       |
| `tests.rs`            |  925 |   ✓    | 单元测试（原内联 `mod tests`）                                           |
| `tone_tiles.rs`       |  543 |        | Low-frequency tone tiles and their bilinear sampling.                    |
| `transforms.rs`       |  563 |        | Per-tile tone transforms.                                                |

#### `panorama_utils/stack_pipeline/virtual_tile.rs`（facade 151 行，7 个子模块）

| 子模块           | 行数 | 仅测试 | 说明                                                                        |
| ---------------- | ---: | :----: | --------------------------------------------------------------------------- |
| `cache_files.rs` |  233 |        | The on-disk meta.json and index.json records of the Virtual_Tile cache.     |
| `cache_key.rs`   |  120 |        | The Virtual_Tile cache key and its source content digest.                   |
| `leases.rs`      |  210 |        | Virtual_Tile residency leases (需求 14.4).                                  |
| `payload_io.rs`  |  166 |        | Lossless payload planes and durable file helpers of the Virtual_Tile cache. |
| `store.rs`       |  773 |        | The Virtual_Tile_Store: lookup, atomic store, verified load and eviction.   |
| `test_access.rs` |   30 |   ✓    | 测试访问入口（原内联 `mod test_access`）                                    |
| `tests.rs`       |  914 |   ✓    | 单元测试（原内联 `mod tests`）                                              |

#### `panorama_utils/stitching.rs`（facade 574 行，34 个子模块）

| 子模块                                          |  行数 | 仅测试 | 说明                                                                       |
| ----------------------------------------------- | ----: | :----: | -------------------------------------------------------------------------- |
| `canvas_fill.rs`                                |   441 |        | Canvas cropping and invalid-margin filling.                                |
| `canvas_maps.rs`                                |   345 |        | Canvas ownership, coverage and confidence maps shared by every compositor. |
| `exposure.rs`                                   |   431 |        | Overlap exposure compensation for progressive and focus compositing.       |
| `filters.rs`                                    |   511 |        | Resampling and filtering primitives.                                       |
| `final_world_owner_runner.rs`                   | 1,067 |   ✓    | Test-only final-world owner oracle runner.                                 |
| `focus_alignment.rs`                            |   909 |        | Full-resolution focus-layer alignment.                                     |
| `focus_analysis.rs`                             |   429 |        | Analysis-scale focus layers and cell measurements.                         |
| `focus_color.rs`                                |   639 |        | Focus-stack colour correction between layers.                              |
| `focus_decision.rs`                             |   304 |        | Focus score maps and per-pixel focus decisions.                            |
| `focus_foreground.rs`                           |   533 |        | Foreground ownership for focus stacks.                                     |
| `focus_fusion_test_access.rs`                   |    15 |   ✓    | 测试访问入口（原内联 `mod focus_fusion_test_access`）                      |
| `focus_layers.rs`                               |   496 |        | Rendered focus layers and Source_RAW sampling for focus stacks.            |
| `focus_masks.rs`                                |   492 |        | Binary masks for focus foreground, background references and placement.    |
| `focus_seam_blend.rs`                           |   489 |        | Focus seam-band blending.                                                  |
| `focus_stack_entry.rs`                          |   415 |        | Focus-stack stitcher entry points and tile render records.                 |
| `focus_stack_stitcher.rs`                       |   972 |        | The global focus-stack compositor body.                                    |
| `focus_tone.rs`                                 |   717 |        | Background tone harmonisation for focus stacks.                            |
| `interpolation_tests.rs（另见下方同名 facade）` |    27 |   ✓    | 单元测试（原内联 `mod interpolation_tests`）                               |
| `layered_compositor.rs`                         |   444 |        | Layered Virtual_Tile compositor entry points and finishing switches.       |
| `panorama_seam_band.rs`                         |   337 |        | Seam-band blending for the progressive panorama stitcher.                  |
| `progressive_seam.rs`                           |   680 |        | Progressive optimal-seam panorama stitcher.                                |
| `sampling.rs`                                   |   161 |        | Homography warping and high-quality source sampling.                       |
| `score_noise_diagnostics.rs`                    |   883 |   ✓    | Test-only score-noise diagnostics recorder.                                |
| `score_noise_probes.rs`                         |   552 |   ✓    | Test-only score-noise probe records and ROI helpers.                       |
| `seam_path.rs`                                  |   388 |        | Minimum-cost seam paths for overlapping sources.                           |
| `station_degradation_wiring_tests.rs`           |   316 |   ✓    | 单元测试（原内联 `mod station_degradation_wiring_tests`）                  |
| `station_seam_apply.rs`                         |   639 |        | Applying station seam plans to the ownership plane.                        |
| `station_seam_diagnostics.rs`                   |   540 |   ✓    | Test-only station seam diagnostics writer.                                 |
| `station_seam_plan.rs`                          |   793 |        | Station seam planning between neighbouring Virtual_Tiles.                  |
| `station_seam_sampling.rs`                      |   539 |        | Final-world sampling for station seams.                                    |
| `station_stroke_match.rs`                       |   491 |   ✓    | Test-only closed-stroke correspondence between station pairs.              |
| `test_tone_capture.rs`                          |   144 |   ✓    | Test-only capture of tone tiles, contour probes and seam grids.            |
| `tile_ownership_stitcher.rs`                    | 1,363 |        | The tile-ownership compositor body shared by every Virtual_Tile path.      |
| `virtual_tiles.rs`                              |   350 |        | Virtual_Tile records and their station geometry.                           |

#### `panorama_utils/stitching/interpolation_tests.rs`（facade 27 行，仅测试，4 个子模块）

| 子模块                 | 行数 | 仅测试 | 说明                                                                       |
| ---------------------- | ---: | :----: | -------------------------------------------------------------------------- |
| `focus_fusion.rs`      |  496 |   ✓    | Focus score, decision, alignment, selection, colour and tone tests for the |
| `geometry_sampling.rs` |  396 |   ✓    | Virtual_Tile geometry, source sampling, focus-layer rendering and          |
| `panorama_blending.rs` |  230 |   ✓    | Overlap exposure, seam-band and pyramid blending tests for the progressive |
| `station_seams.rs`     |  351 |   ✓    | Station stroke matching and station seam planning tests.                   |

#### `panorama_virtual_tile_tests.rs`（facade 310 行，仅测试，12 个子模块）

| 子模块                       | 行数 | 仅测试 | 说明                                                                                           |
| ---------------------------- | ---: | :----: | ---------------------------------------------------------------------------------------------- |
| `coverage_measurements.rs`   |  428 |   ✓    | Coverage-aware measurements: requirement 6.8 on covered pixels, directly and through the       |
| `fine_polish.rs`             |  358 |   ✓    | The fine polish stage: NCC and best window offsets, low-texture templates, projective polish   |
| `global_search.rs`           |  455 |   ✓    | Prior-free global search: similarity proposals against an independent brute-force oracle,      |
| `hypothesis_selection.rs`    |  358 |   ✓    | Multi-hypothesis Station_Relation selection: a compliant model when the projective one fails   |
| `match_pyramid.rs`           |  392 |   ✓    | Match planes and the pyramid: pyramid factors, block-mean match planes without uncovered       |
| `prior_repair_planning.rs`   |  639 |   ✓    | Rendering-prior repair planning and the authoritative relation pairs it starts from: seed      |
| `raw_correlation.rs`         |  137 |   ✓    | Source_RAW correlation proposals: large translations, repeated-texture peaks, uncovered        |
| `raw_direct_refinement.rs`   |  486 |   ✓    | Source_RAW direct photometric refinement: reciprocal multi-peak pairing, a brute-force oracle, |
| `raw_independent_support.rs` |  217 |   ✓    | Source_RAW independent support: per-pair spatial proposals with match provenance, proposal     |
| `residual_correction.rs`     |  231 |   ✓    | Bounded residual correction and residual model selection: local scale over the evidence        |
| `scan_grid_closure.rs`       |  301 |   ✓    | Checkpoint 10 closure on a synthetic two-dimensional scan grid: tree drift, cross-station      |
| `station_records.rs`         |  158 |   ✓    | Intra-station frame records and Station_Relation record compatibility: a re-measured frame     |

#### `raw_processing_acceptance.rs`（facade 355 行，仅测试，4 个子模块）

| 子模块                 | 行数 | 仅测试 | 说明                                                          |
| ---------------------- | ---: | :----: | ------------------------------------------------------------- |
| `fixture_run.rs`       |  395 |   ✓    | One manifest fixture through the RAW pipeline.                |
| `node_probe.rs`        |  255 |   ✓    | The centered CFA probe and its rawler develop stages.         |
| `sampling.rs`          |  127 |   ✓    | Strided sample summaries of RAW mosaics and developed images. |
| `scenario_coverage.rs` |   61 |   ✓    | Sony α7R V scenario coverage of the acceptance manifest.      |

#### `style_transfer.rs`（facade 182 行，4 个子模块）

| 子模块              | 行数 | 仅测试 | 说明                                                   |
| ------------------- | ---: | :----: | ------------------------------------------------------ |
| `color_transfer.rs` |  339 |        | Colour statistics and the transfer transform.          |
| `preview_source.rs` |  223 |        | Source loading and bounded previews for the Style Lab. |
| `styled_output.rs`  |  206 |        | Writing the styled result.                             |
| `tests.rs`          |  161 |   ✓    | 单元测试（原内联 `mod tests`）                         |

### 前端：文件与同名目录下的 parts

#### `src/App.tsx`（983 行，1 个 part）

| part            | 行数 | 导出                     |
| --------------- | ---: | ------------------------ |
| `folderTree.ts` |   23 | `insertChildrenIntoTree` |

#### `src/components/adjustments/Curves.tsx`（831 行，3 个 part）

| part                 | 行数 | 导出                                                                                                                                                                     |
| -------------------- | ---: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `histogramPaths.ts`  |   17 | `getHistogramPath`, `getZeroHistogramPath`                                                                                                                               |
| `parametricCurve.ts` |  108 | `DEFAULT_PARAMETRIC_CURVE_SETTINGS`, `DEFAULT_PARAMETRIC_CURVE`, `buildParametricPoints`, `isDefaultParametricCurve`, `getSplitterGradient`, `convertParametricToPoints` |
| `pointCurve.ts`      |  104 | `DEFAULT_POINT_CURVES`, `getCurvePath`, `isDefaultCurve`                                                                                                                 |

#### `src/components/modals/BatchGeometryModal.tsx`（762 行，3 个 part）

| part                          | 行数 | 导出                                                                                                                                                                                   |
| ----------------------------- | ---: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `BatchGeometryComparison.tsx` |  497 | `BatchGeometryComparison`                                                                                                                                                              |
| `types.ts`                    |   37 | `BatchOutputFormat`, `ProcessingPhase`, `BatchGeometryModalProps`, `BatchGeometryProgress`, `BatchGeometryPreviewItem`, `BatchGeometryAnalysisResult`, `BatchGeometryCorrectionResult` |
| `utils.ts`                    |   33 | `normalizeExtensions`, `pathsFromSelection`, `displayFileName`, `getSupportedImageTypes`, `errorMessage`, `ORIENTATION_KEYS`, `normalizeOrientationSteps`                              |

#### `src/components/modals/StyleTransferModal.tsx`（795 行，4 个 part）

| part             | 行数 | 导出                                                                                                                    |
| ---------------- | ---: | ----------------------------------------------------------------------------------------------------------------------- |
| `SourceCard.tsx` |   91 | `SourceCard`                                                                                                            |
| `constants.ts`   |   57 | `MAX_PREVIEW_EDGE`, `DEFAULT_STRENGTH`, `IMAGE_EXTENSIONS`                                                              |
| `types.ts`       |   22 | `SourceRole`, `StyleImage`, `ImageDimensions`, `StyleTransferModalProps`                                                |
| `utils.ts`       |   51 | `readImageElement`, `getFileName`, `getOutputName`, `orientDimensionsLikePreview`, `makeImageData`, `toNativeImageData` |

#### `src/components/panel/SettingsPanel.tsx`（1,721 行，5 个 part）

| part               | 行数 | 导出                                                                                                                                           |
| ------------------ | ---: | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `KeybindRow.tsx`   |   85 | `KeybindRow`                                                                                                                                   |
| `ModeSwitches.tsx` |   91 | `CanvasInputModeSwitch`, `PreviewModeSwitch`                                                                                                   |
| `SettingItems.tsx` |   68 | `SettingItem`, `DataActionItem`                                                                                                                |
| `constants.ts`     |   47 | `THIRD_PARTY_DISPLAY_NAMES`, `EXECUTE_TIMEOUT`, `adjustmentVisibilityDefaults`, `resolutions`, `thumbnailResolutions`, `zoomMultiplierOptions` |
| `types.ts`         |   23 | `ConfirmModalState`, `SettingsPanelProps`, `MyLens`                                                                                            |

#### `src/components/panel/editor/ImageCanvas.tsx`（2,246 行，4 个 part）

| part              | 行数 | 导出                                             |
| ----------------- | ---: | ------------------------------------------------ |
| `BrushLines.tsx`  |   83 | `OptimizedBrushLine`, `SourcePreviewLine`        |
| `MaskOverlay.tsx` |  971 | `MaskOverlay`                                    |
| `edgeFade.ts`     |   12 | `getEdgeFadeStyle`                               |
| `types.ts`        |   75 | `CursorPreview`, `DrawnLine`, `ImageCanvasProps` |

#### `src/components/panel/library/CullingView.tsx`（274 行，3 个 part）

| part                 | 行数 | 导出                    |
| -------------------- | ---: | ----------------------- |
| `CullingPreview.tsx` |  918 | `CullingPreview`        |
| `CullingRow.tsx`     |   60 | `VirtualizedCullingRow` |
| `types.ts`           |    6 | `SyncViewport`          |

#### `src/components/panel/right/AIPanel.tsx`（965 行，7 个 part）

| part                   | 行数 | 导出                                          |
| ---------------------- | ---: | --------------------------------------------- |
| `BrushTools.tsx`       |   52 | `BrushTools`                                  |
| `ConnectionStatus.tsx` |   27 | `ConnectionStatus`                            |
| `ContainerRow.tsx`     |  324 | `ContainerRow`                                |
| `DragAndDrop.tsx`      |   99 | `DragData`, `AiListRoot`, `DraggableGridItem` |
| `SettingsPanel.tsx`    |  202 | `SettingsPanel`                               |
| `SubMaskRow.tsx`       |  246 | `SubMaskRow`                                  |
| `subMaskConfig.ts`     |   35 | `SUB_MASK_CONFIG`                             |

#### `src/components/panel/right/CropPanel.tsx`（1,452 行，3 个 part）

| part                 | 行数 | 导出                                                                           |
| -------------------- | ---: | ------------------------------------------------------------------------------ |
| `analysisResults.ts` |   15 | `StraightenAnalysisResult`, `UprightAnalysisResult`                            |
| `aspectRatios.ts`    |   18 | `BASE_RATIO`, `ORIGINAL_RATIO`, `RATIO_TOLERANCE`, `orientRatio`, `CropPreset` |
| `cropSnapshots.ts`   |   53 | `CropSessionSnapshot`, `createCropSessionSnapshot`, `createCropResetSnapshot`  |

#### `src/components/panel/right/ExportPanel.tsx`（905 行，2 个 part）

| part                   | 行数 | 导出               |
| ---------------------- | ---: | ------------------ |
| `Section.tsx`          |   18 | `Section`          |
| `WatermarkPreview.tsx` |  118 | `WatermarkPreview` |

#### `src/components/panel/right/FolderTree.tsx`（505 行，6 个 part）

| part                 | 行数 | 导出                                                                                                                  |
| -------------------- | ---: | --------------------------------------------------------------------------------------------------------------------- |
| `AlbumTreeNode.tsx`  |  138 | `AlbumTreeNode`                                                                                                       |
| `FolderSortMenu.tsx` |  118 | `FolderSortMenu`                                                                                                      |
| `SectionHeader.tsx`  |   28 | `SectionHeader`                                                                                                       |
| `TreeNode.tsx`       |  194 | `VisibleProps`, `TreeNode`                                                                                            |
| `albumIcons.ts`      |   15 | `ALBUM_ICONS`                                                                                                         |
| `treeUtils.ts`       |   88 | `FolderTree`, `filterTree`, `getAutoExpandedPaths`, `filterAlbumTree`, `getAutoExpandedAlbumGroups`, `sortFolderTree` |

#### `src/components/panel/right/MasksPanel.tsx`（999 行，6 个 part）

| part                | 行数 | 导出                                                                |
| ------------------- | ---: | ------------------------------------------------------------------- |
| `BrushTools.tsx`    |   96 | `BrushTools`, `FlowBrushTool`                                       |
| `ContainerRow.tsx`  |  334 | `ContainerRow`                                                      |
| `DragAndDrop.tsx`   |   94 | `DragData`, `MasksListRoot`, `NewMaskDropZone`, `DraggableGridItem` |
| `SettingsPanel.tsx` |  427 | `SettingsPanel`                                                     |
| `SubMaskRow.tsx`    |  272 | `SubMaskRow`                                                        |
| `subMaskConfig.ts`  |   49 | `SUB_MASK_CONFIG`                                                   |

#### `src/components/panel/right/PresetsPanel.tsx`（897 行，3 个 part）

| part                    | 行数 | 导出                                                                                   |
| ----------------------- | ---: | -------------------------------------------------------------------------------------- |
| `DragAndDrop.tsx`       |  222 | `FolderItemDisplay`, `DraggablePresetItem`, `RootDroppableArea`, `DroppableFolderItem` |
| `PresetItemDisplay.tsx` |  178 | `PresetItemDisplay`                                                                    |
| `mixAdjustments.ts`     |   91 | `mixAdjustments`                                                                       |

#### `src/utils/adjustments.ts`（212 行，5 个 part）

| part             | 行数 | 导出                                                                                                                                                                                                                                      |
| ---------------- | ---: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `colorLabels.ts` |   12 | `Color`, `COLOR_LABELS`                                                                                                                                                                                                                   |
| `defaults.ts`    |  240 | `DEFAULT_PARAMETRIC_CURVE_SETTINGS`, `getDefaultParametricCurve`, `getDefaultCurves`, `DEFAULT_PARAMETRIC_CURVE`, `INITIAL_MASK_ADJUSTMENTS`, `INITIAL_MASK_CONTAINER`, `INITIAL_ADJUSTMENTS`                                             |
| `enums.ts`       |  118 | `ActiveChannel`, `DisplayMode`, `PasteMode`, `BasicAdjustment`, `ColorAdjustment`, `ColorGrading`, `DetailsAdjustment`, `Effect`, `CreativeAdjustment`, `TransformAdjustment`, `LensAdjustment`                                           |
| `sections.ts`    |  278 | `AdjustmentGroup`, `ADJUSTMENT_GROUPS`, `COPYABLE_ADJUSTMENT_KEYS`, `ADJUSTMENT_SECTIONS`, `CAMERA_RAW_ADJUSTMENT_SECTIONS`                                                                                                               |
| `types.ts`       |  260 | `CopyPasteSettings`, `ColorCalibration`, `ParametricCurveSettings`, `ParametricCurve`, `Adjustments`, `AiPatch`, `ColorGradingProps`, `Coord`, `Curves`, `HueSatLum`, `MaskAdjustments`, `MaskContainer`, `Sections`, `SectionVisibility` |

#### `src/styles.css`（27 行，按顺序 `@import` 13 个文件）

| 文件                                     | 行数 |
| ---------------------------------------- | ---: |
| `src/styles/tokens.css`                  |  148 |
| `src/styles/app/base.css`                |  167 |
| `src/styles/app/core-components.css`     |  760 |
| `src/styles/app/develop-panel.css`       |  557 |
| `src/styles/app/camera-raw-controls.css` |  597 |
| `src/styles/app/crop-panel.css`          |  702 |
| `src/styles/app/editor-chrome.css`       |  797 |
| `src/styles/app/controls.css`            |  402 |
| `src/styles/app/style-transfer.css`      |  913 |
| `src/styles/app/modals.css`              |  756 |
| `src/styles/app/editor-popovers.css`     |  290 |
| `src/styles/app/app-surfaces.css`        |  391 |
| `src/styles/app/develop-controls.css`    |  514 |

### 仍超过 1000 行的源文件

| 文件                                                                 |  行数 |
| -------------------------------------------------------------------- | ----: |
| `src-tauri/src/panorama_stitching/stitch_images.rs`                  | 3,926 |
| `src/components/panel/Editor.tsx`                                    | 2,463 |
| `src/components/panel/editor/ImageCanvas.tsx`                        | 2,246 |
| `src/components/panel/SettingsPanel.tsx`                             | 1,721 |
| `tests/focus-stack-quality-contract.mjs`                             | 1,691 |
| `src/components/panel/right/CropPanel.tsx`                           | 1,452 |
| `src/hooks/useAppContextMenus.ts`                                    | 1,366 |
| `src-tauri/src/panorama_utils/stitching/tile_ownership_stitcher.rs`  | 1,363 |
| `src-tauri/src/panorama_utils/stitching/final_world_owner_runner.rs` | 1,067 |

<!-- source-layout:end -->

## 仍超过 1000 行的文件

下表的文件按 2026-10-29 的决定不再拆：它们的主体是单个函数、组件或 hook，继续缩短需要改写函数内部，超出“只移动代码”
的范围。

| 文件                                                                 | 主体                                                     | 说明                                                                         |
| -------------------------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `src-tauri/src/panorama_stitching/stitch_images.rs`                  | `stitch_images_with_options`，约 3,820 行                | 全景/焦点堆栈的总编排函数，若要缩短需按阶段拆函数并单独核对行为              |
| `src-tauri/src/panorama_utils/stitching/tile_ownership_stitcher.rs`  | `focus_tile_ownership_stitcher_with_finishing`，1,302 行 | 分层 Virtual_Tile 的 ownership 合成主体                                      |
| `src-tauri/src/panorama_utils/stitching/final_world_owner_runner.rs` | `run_final_world_owner_oracle`，1,049 行                 | 仅测试的研究 oracle 驱动                                                     |
| `src/components/panel/Editor.tsx`                                    | `Editor` 组件，2,387 行                                  | 另因旁边已有 `editor/` 目录，大小写冲突，不新建 `Editor/`                    |
| `src/components/panel/editor/ImageCanvas.tsx`                        | `ImageCanvas` 组件，2,225 行                             | 其余声明已移到 `ImageCanvas/`                                                |
| `src/components/panel/SettingsPanel.tsx`                             | `SettingsPanel` 组件，1,686 行                           | 其余声明已移到 `SettingsPanel/`                                              |
| `src/components/panel/right/CropPanel.tsx`                           | `CropPanel` 组件，1,398 行                               | 其余声明已移到 `CropPanel/`                                                  |
| `src/hooks/useAppContextMenus.ts`                                    | `useAppContextMenus` hook，1,293 行                      | 文件中没有其他可移动的声明                                                   |
| `tests/focus-stack-quality-contract.mjs`                             | 源码契约脚本                                             | 断言共享同一组源码变量；按 N2 计划，以后按“生产不变量 / 研究 oracle”语义拆分 |
