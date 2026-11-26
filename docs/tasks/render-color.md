# 渲染、内存与色彩任务（RM / CL）

> [返回总入口](../project-progress.md) · 书画堆栈开发见 [tasks.md](../../.kiro/specs/layered-camera-group-focus-stitching/tasks.md)

2026-11-12从`docs/project-progress.md` §5原样迁出（链接已按新位置改写）；以后在本文件勾选和追加。

<a id="render"></a>

## 渲染与内存（RM）

细节见[渲染层级与内存合同](../render-strategy.md)。以下实现状态来自既有专项记录，未在本轮文档整理重测。

- [x] **RM.1** 四级渲染/ROI与缓存合同、共享图像所有权、RAW开发结果RGB32F收敛、逐带GPU上传及桌面JPEG/PNG/TIFF流式输出已记录验证；普通/正交几何、范围及AI蒙版halo等优化有定向基准。
- [ ] **RM.2** 精细旋转/镜头模糊的二次重采样和非浮点路径继续降低整图物化；按真实profile决定实现顺序。
- [ ] **RM.3** rawler完整mosaic/scratch/开发帧的流式合同；先评估上游接口或替代方案，不能仅缩小缓存就声称完整RAW开发已分块。
- [ ] **RM.4** WebP整图CPU输入，以及JXL/AVIF/Android完整编码图与压缩缓冲峰值；评估可替换分块编码器并保留色彩/alpha/错误合同。
- [ ] **RM.5** 最终全尺寸蒙版bitmap、RAW默认显示匹配源和AI解码源的存储/消费合同优化；已有tile scratch收敛不等于最终驻留源消失。
- [ ] **RM.6** 接入系统/驱动实时内存压力；保留当前高水位回收。若改panic=abort，先处理MozJPEG FFI错误边界。
- [ ] **RM.7** 应用内60MP首屏、滑块P95、100% GPU ROI、GPU文件导出与进程峰值实测；CPU样片与合成基准不能替代。

## 色彩与相机实测（CL）

细节见[色彩管线契约](../color-pipeline.md)及[α7R V范围化基线](../../tests/acceptance/sony-a7rv-60mp-baseline.md)。

- [x] **CL.1** CPU/WGSL传递函数统一、RAW/普通图编码语义、RGB输入ICC转换、核心预览/导出sRGB ICC与原生SDR surface边界已建立契约检查。
- [x] **CL.2** CMYK/Gray、CICP-only、无效/超限profile的兼容边界和测试（2026-11-27 `a6e1f103` 完成：`color_management.rs` 与 `image_loader/source_decode.rs` 已实现 ICC 头部显式分类、4 MiB 上限与损坏 tag 表不改写像素保护、灰度图内嵌 Gray ICC 归一化、无 `iCCP` 全范围 SDR RGB PNG `cICP` 归一化，以及 CMYK/HDR/窄范围/错配 Profile 安全回退 sRGB；单测与端到端解码测试 21 passed，`color-contract:check` 25/25 通过）；自定义显示profile、软打样及纸墨模拟未提供，列为后续能力评估。
- [x] **CL.3** 旧HDR/全景等非basic预览链路的ICC统一；线性节点参考图及WebView/WGPU/各导出格式像素一致性补测（2026-11-27 完成：在 `color_management.rs` 集中提供 `encode_srgb_png_bytes`、`encode_srgb_jpeg_bytes`、`save_srgb_tagged_image`，统一旧 HDR、旧全景、降噪前后对比、批量几何校正、底片反转共 5 处非基础 WebView 预览及同批编码 sRGB 保存的 sRGB v4 ICC 嵌入，线性 RAW 降噪 TIFF 明确保持未标 sRGB；新增线性节点合成色卡单测、WebView/PNG/TIFF/JPEG/WebP/无损 JXL 跨格式像素与 ICC 一致性回归（38 passed / 0 failed）、本机 Metal WGPU `is_raw=0/1` 与 `+1 EV` 色彩回读实测（1 passed，误差 ≤1 LSB），并接入 `color-contract-boundary.mjs` 静态契约保护；勘察设计见 [CL.3 勘察与改造设计](cl3-preview-icc-design.md)）。
- [x] **CL.4** 单个Sony α7R V有损ARW长曝光样片的范围化CPU质量与60MP性能基线建立，保留CPU默认输出高光/影调差异。
- [ ] **CL.5** 日光、钨丝灯、高ISO、欠曝、高饱和光、lossless-L、uncompressed样片及色卡/配对参考；未具备证据前不宣称绝对色准、ACR等价或全相机质量通过。
