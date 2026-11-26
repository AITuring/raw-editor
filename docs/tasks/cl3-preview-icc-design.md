# CL.3 勘察与改造设计：非基础预览/保存链路 ICC 统一与跨格式像素一致性

> [返回渲染与色彩任务](render-color.md) · [色彩管线契约](../color-pipeline.md)

更新日期：2026-11-27（零 Rust 编译勘察阶段，基准提交 `a6e1f103`）。

## 1. 目标范围

对应 [render-color.md](render-color.md) 的 **CL.3**：

1. 统一旧 HDR、旧全景、降噪、批量几何校正、底片反转等**非 basic 预览链路**的 sRGB v4 ICC 嵌入，消除 WebView 依赖浏览器默认未标记假设。
2. 同步收口同批非基础保存路径的编码 sRGB ICC 写入（严格区分线性 RAW 数据，不得误标 sRGB ICC）。
3. 补齐**线性节点参考图测试**与 **WebView / WGPU / 各导出格式像素一致性回归测试**，并纳入 `tests/color-contract-boundary.mjs`。

## 2. 全仓库编码点逐行勘察清单

### 2.1 缺失 sRGB ICC 的 WebView 预览链路（共 5 处，必须统一）

| 模块                 | 文件与位置                                                        | 当前实现                                                                      | 问题                                                                | 改造目标                                        |
| -------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------- | ----------------------------------------------- |
| 旧 HDR 合并预览      | `src-tauri/src/lib/merged_outputs.rs:54-60`                       | `hdr_merged.to_rgb8().write_to(&mut buf, ImageFormat::Png)`                   | 未写入 `iCCP` sRGB 块                                               | 改为 `color_management::encode_srgb_png_bytes`  |
| 旧全景拼接预览       | `src-tauri/src/panorama_stitching.rs:863-872`                     | `preview_u8.write_to(&mut buf, ImageFormat::Png)`                             | 未写入 `iCCP` sRGB 块（新版 `image_stack/preview_files.rs` 已嵌入） | 改为 `color_management::encode_srgb_png_bytes`  |
| 降噪前后对比预览     | `src-tauri/src/denoising/pipeline.rs:164-189`                     | `denoised_preview` 与 `original_preview` 均 `write_to(..., ImageFormat::Png)` | 两组对比图均缺 `iCCP`                                               | 改为 `color_management::encode_srgb_png_bytes`  |
| 批量几何校正前后预览 | `src-tauri/src/export_processing/batch_geometry_preview.rs:37-47` | `JpegEncoder::new_with_quality(&mut bytes, 90)`                               | 未调用 `set_icc_profile(srgb_v4_profile())`                         | 改为 `color_management::encode_srgb_jpeg_bytes` |
| 底片反转实时预览     | `src-tauri/src/negative_conversion.rs:277-284`                    | `JpegEncoder::new_with_quality(&mut buf, 80)`                                 | 未调用 `set_icc_profile(srgb_v4_profile())`                         | 改为 `color_management::encode_srgb_jpeg_bytes` |

### 2.2 缺失 sRGB ICC 的非基础保存路径（共 5 处，区分编码域与线性域）

| 模块                                                      | 文件与位置                                                   | 当前实现                                                                                                           | 色彩空间语义                                                             | 改造策略                                                                                                    |
| --------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| 旧 HDR 保存 (`save_hdr`)                                  | `src-tauri/src/lib/merged_outputs.rs:110-112`                | `image_to_save.save(&output_path)`                                                                                 | 经 `apply_linear_to_srgb` 后为编码 sRGB（PNG / RGB32F TIFF）             | PNG 经 `save_srgb_tagged_image` 嵌入 sRGB ICC；RGB32F TIFF 如写入 sRGB 编码值则同步写 `TiffTag::IccProfile` |
| 旧全景保存 (`save_panorama`)                              | `src-tauri/src/panorama_stitching.rs:925-937`                | `.save(&output_path)`                                                                                              | 编码 sRGB PNG / TIFF                                                     | 经 `save_srgb_tagged_image` 嵌入 sRGB ICC                                                                   |
| 降噪保存 (`save_denoised_image` / `batch_denoise_images`) | `src-tauri/src/denoising.rs:161, 249-251`                    | `image_to_save.save(&output_path)`                                                                                 | **注意区分**：`!is_raw` 输出编码 sRGB PNG；`is_raw` 输出线性 16-bit TIFF | 仅 `!is_raw` PNG 嵌入 sRGB ICC；**严禁**对线性 RAW TIFF 误标 sRGB ICC                                       |
| 风格迁移导出 (`styled_output.rs`)                         | `src-tauri/src/style_transfer/styled_output.rs:35-37, 85-87` | 桌面 JPEG 已通过 `encode_streaming_jpeg(..., true, ...)` 嵌 ICC；PNG 走 `.save_with_format`，Android JPEG 未设 ICC | 编码 sRGB                                                                | PNG 与 Android JPEG 改为通过 `color_management` 写入 sRGB v4 ICC                                            |
| 底片反转导出 (`convert_negatives`)                        | `src-tauri/src/negative_conversion.rs:329-332`               | `processed.to_rgb16().save(&out_path)`                                                                             | 编码 sRGB 16-bit 正片 TIFF                                               | 经 `save_srgb_tagged_image`（`TiffEncoder::set_icc_profile`）写入 sRGB v4 ICC                               |

### 2.3 已核对合规或明确豁免的路径（不纳入修改）

- **已嵌入 sRGB v4 ICC**：核心编辑预览（`srgb_preview_encoder`）、Style Transfer 预览（`style_transfer.rs:79`）、LUT 色卡预览（`lut_processing.rs`）、图库缩略图（`thumbnail_render.rs:420`）、现代 Image Stack 预览与导出（`image_stack/preview_files.rs`、`image_stack/output_encoding.rs`）、主导出链路 JPEG/PNG/TIFF/WebP/JXL（`export_processing/`）。
- **明确豁免（非色彩预览）**：
  - `mask_generation.rs:247`、`ai_commands.rs:20,200`、`encode_grayscale_to_png`：单通道蒙版/透明度数据平面，不应附带色彩 ICC。
  - `inpainting.rs:294,303,508,517`：内部修补 patch 与灰度蒙版传输，显式携带 `isSrgbEncoded` 标志供 GPU 管线消费，不直接作为色彩管理视图展示。
  - `export_processing/encode_bytes.rs:192-196`（AVIF）：`image` crate 当前 `AvifEncoder` 未暴露 `set_icc_profile`，依赖 `ravif` 默认 BT.709/sRGB 编码并在文档中显式记录边界。

## 3. 共享 Rust API 设计（零重复编码实现）

在 `src-tauri/src/color_management.rs` 中集中新增 3 个小而纯粹的公共辅助函数：

```rust
/// 将编码 sRGB 图像编码为嵌入标准 sRGB v4 ICC 的 PNG 字节（供 WebView 预览复用）。
pub fn encode_srgb_png_bytes(image: &DynamicImage) -> Result<Vec<u8>, String>;

/// 将编码 sRGB 图像编码为嵌入标准 sRGB v4 ICC 的 JPEG 字节（供 WebView 预览复用）。
pub fn encode_srgb_jpeg_bytes(image: &DynamicImage, quality: u8) -> Result<Vec<u8>, String>;

/// 将编码 sRGB 图像按目标后缀（PNG / JPEG / TIFF）写入文件并嵌入 sRGB v4 ICC。
pub fn save_srgb_tagged_image(image: &DynamicImage, path: &Path) -> Result<(), String>;
```

约束：

- 仅用于**已完成 OETF 的编码 sRGB 图像**；线性 RAW 数据禁止调用。
- 复用唯一的 `srgb_v4_profile()`，不引入任何第二套 ICC 字节或重复传递函数常量。

## 4. 线性节点参考图与跨格式像素一致性测试矩阵

### 4.1 线性节点合成色卡单测（纯 CPU，CI 默认运行）

在 `src-tauri/src/color_management.rs` 新增确定性合成色卡（覆盖纯黑 `0.0`、18% 线性灰 `0.18`、暗部线性段 `0.002`、高饱和 Display P3 原色与中性灰阶）：

1. Display P3 编码像素 → `normalize_encoded_rgb_profile_to_srgb` → `srgb_to_linear_channel` → 验证线性节点与参考值误差 ≤ `1e-4`。
2. 线性曝光 `+1 EV`（`×2.0`）与 `-1 EV`（`×0.5`）在线性域可逆（未钳制范围内误差 ≤ `1e-6`），再经 `linear_to_srgb_channel` 量化至 8-bit / 16-bit 验证无暗部色带与传递函数漂移。

### 4.2 跨格式像素与 ICC 往返一致性单测（纯 CPU，CI 默认运行）

在 `src-tauri/src/export_processing/tests/encoded_formats.rs` 新增统一参考图测试：将同一张覆盖灰阶与彩色色块的 `DynamicImage` 分别送入：

- WebView JPEG 预览（`srgb_preview_encoder`）
- WebView PNG 预览（`encode_srgb_png_bytes`）
- WebView JPEG 预览（`encode_srgb_jpeg_bytes`）
- 导出 `PNG`（8/16-bit）、`TIFF`（8/16-bit）、`JPEG`（q=100）、`WebP`（q=100）、`JXL`（q=100 无损）

逐一经 `image_loader` / `ImageReader` 读回并断言：

1. 每种格式均携带与 `srgb_v4_profile()` 逐字节一致的 ICC Profile。
2. 无损格式（PNG 预览、PNG 导出、TIFF 导出、无损 JXL）解码后逐像素 RGB 差值 **严格为 0 LSB**。
3. 有损格式（WebView JPEG、导出 JPEG q=100、WebP q=100）在均匀色块中心区域通道最大偏差不超过编解码量化界（`<= 2` LSB），且零色域/Gamma 偏移。

### 4.3 WGPU 渲染回读一致性测试（`#[ignore]` GPU 门禁，沿用现有 `gpu_processing` 约定）

在 `gpu_processing` 测试中增加合成色卡单测：

- 分别以 `is_raw = 0`（sRGB 编码输入）与 `is_raw = 1`（线性 sRGB 输入）运行无调整（neutral）主 shader；
- 校验 `RGBA8Unorm` 回读像素与 CPU 线性参考管线逐通道误差 `<= 1` LSB，确保 WebView 预览、原生 WGPU `SurfaceColorSpace::Srgb` 与导出字节消费同一输出合同。

### 4.4 静态契约保护（`tests/color-contract-boundary.mjs`）

新增断言锁定 5 处非基础预览模块必须调用 `encode_srgb_png_bytes` / `encode_srgb_jpeg_bytes`，禁止回退为裸 `write_to(..., ImageFormat::Png)` 或未附带 `set_icc_profile` 的 `JpegEncoder::new_with_quality`。

## 5. 执行顺序与并行冲突评估

- **零冲突确认**：所有目标文件均不涉及 `2B.14` 正在修改的 `stitching/focus_layers.rs` 与 `interpolation_tests`（仅 `panorama_stitching.rs` 尾部旧 `stitch_panorama`/`save_panorama` 命令与 `2B.14` worktree 路径完全独立）。
- **待 `2B.14` 释放 `build` 槽后落地**：
  1. 先验收并提交已就绪的 **Task 20 Phase 3**（Rust + 前端导出包审计）；
  2. 再按本设计一次性落地 **CL.3** 的 3 个共享辅助函数、5 处预览 + 5 处非基础保存迁移、线性节点 + 跨格式像素一致性单测与 `color-contract-boundary.mjs` 契约更新。
