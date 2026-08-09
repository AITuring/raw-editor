# 渲染层级与内存合同

本文记录 RAW Editor M1 阶段的预览/导出分辨率层级、原生缓冲所有权和确定性内存基线。
这里的 9504×6336 只代表约 60MP 的尺寸算术，不是 Sony α7R V 真实样片测试，也不用于声明真实
解码速度、画质或进程峰值内存。

## 四级渲染策略

| 层级                   | 触发条件                                       | 分辨率合同                                                                                 | 输出合同                                                                               |
| ---------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| `rapidPreview`         | 滑块或其他连续交互进行中                       | `full` 保留请求尺寸；`high` 不超过大图半分辨率；`performance` 进一步降至半分辨率预算的 75% | 可见区域使用二进制 JPEG patch；原生 WGPU 可直接更新 surface                            |
| `halfResolutionEdit`   | 交互停止、尚未达到源像素级观察                 | 上限为基础预览尺寸与源尺寸一半中的较大者，但不超过源尺寸；小图不被强制缩半                 | 完整预览或带 overscan 的 settled ROI                                                   |
| `fullResolutionRoi`    | 交互停止、视口请求已达到源像素分辨率且存在 ROI | 输入最长边等于源图，GPU/CPU 只回读可见 ROI                                                 | ROI 使用 JPEG 4:4:4、质量 100 的二进制 patch                                           |
| `fullResolutionExport` | 文件导出                                       | 变换与调整在完整输入尺寸执行；用户要求的 resize 和 watermark 只在处理完成后应用            | 无 resize/watermark 的 PNG/TIFF 可逐带写文件；其他情况走完整帧原生编码，不经过 WebView |

前端 `resolvePreviewRenderPlan` 负责选择前三个层级并传递 camelCase `renderTier`；Rust
`resolve_preview_render_tier` 再次验证交互状态、ROI 和源分辨率，拒绝把
`fullResolutionExport` 送入预览 worker。导出管线固定标记为 `fullResolutionExport`，不会复用预览降采样。

所有层级继续使用同一套调整参数、WGSL 和 sRGB 合同；层级只改变处理范围、分辨率、传输和缓存，
不允许使用导出无法复现的临时图像算法。

## 缓冲所有权收敛

本阶段收敛了七类可确定的重复缓冲或高水位资源：

1. `CachedPreview` 只保存当前层级的一张 `Arc<DynamicImage>`。快速预览直接生成目标尺寸，
   不再先生成较大底图后再长期保存一张 `small_image`。
2. 无蒙版任务绑定复用的 1×1 `R8Unorm` dummy texture，不再构造两层全尺寸零数组。
   有蒙版任务按实际活动层数创建 texture，并从现有 bitmap 逐层 `queue.write_texture`，不再把所有
   bitmap 拼成第二份连续 CPU 数组。缺失 bitmap 的已声明层由 WGPU 的资源初始化保持为零。
3. WebView 回读的 RGBA8 结果使用 `Arc<DynamicImage>` 同时交给 JPEG 编码和分析 worker，
   不再为 histogram/waveform 调用 `processed_pixels.clone()`。
4. 无 patch、几何、旋转、翻转、镜头模糊或裁剪时，全尺寸变换缓存直接复用已解码图像的 `Arc`，
   不再因 `Cow::into_owned()` 复制一张完整 RGB32F 图。
5. 无 resize、无 watermark 的 PNG/TIFF 导出按横向 tile 完成一个最多 2048 行的带状 RGBA8 缓冲后，
   立刻逐行交给编码器；不再构造完整尺寸的最终 `DynamicImage`。输出先写同目录临时文件，完整编码、
   元数据处理和取消检查都成功后才原子替换目标路径。
6. CPU 回读任务只保留 tile 工作 texture；原生显示所需的 `working_texture` 与 `output_texture` 在
   该 processor 中降为 1×1。大尺寸显示 processor 转入 CPU 导出时会按 512 MiB 高水位和滞回条件收缩。
7. 批量任务全部 join 后，若 processor 与 RGBA16F 输入 texture 的逻辑占用合计达到 512 MiB，立即
   释放两者并触发一次有界 GPU poll，避免 60MP 导出的高水位常驻到后续编辑。

## 确定性前后基线

下表以 9504×6336（60,217,344 像素）计算应用代码可见的逻辑像素缓冲。MiB 使用 1,048,576
字节；这些数字不包含驱动对齐、allocator 元数据、编码器内部状态或操作系统 RSS。

| 缓冲                                                   |                     修改前 |                   修改后 |          减少 |
| ------------------------------------------------------ | -------------------------: | -----------------------: | ------------: |
| 无蒙版 CPU 两层零上传数组                              | 120,434,688 B（114.9 MiB） |                      0 B | 120,434,688 B |
| 无蒙版 GPU `R8` 两层 texture                           | 120,434,688 B（114.9 MiB） |                1 B dummy | 120,434,687 B |
| WebView 分析用 RGBA8 像素复制                          | 240,869,376 B（229.7 MiB） |        0 B（`Arc` 共享） | 240,869,376 B |
| 未变换 RGB32F 全尺寸缓存复制                           | 722,608,128 B（689.1 MiB） |        0 B（`Arc` 共享） | 722,608,128 B |
| 旧 Performance 1.5× 降采样的额外 RGB32F `small_image`  | 321,159,168 B（306.3 MiB） |      0 B（单一目标底图） | 321,159,168 B |
| PNG/TIFF 最终 RGBA8 CPU 编码缓冲                       | 240,869,376 B（229.7 MiB） | 77,856,768 B（74.2 MiB） | 163,012,608 B |
| CPU 导出 processor 的两张 9728×6400 RGBA8 显示 surface | 498,073,600 B（475.0 MiB） |          8 B（两张 1×1） | 498,073,592 B |

这些减少项不能简单相加当作真实进程峰值：它们的生命周期并不完全重叠。回归测试只锁定每一项
不再由应用代码重新分配。

## 自动验证

- `npm run preview-resolution:check`：验证 Retina 100%、快速预览、半分辨率编辑和全分辨率 ROI 选择。
- `npm run render-strategy:check`：锁定四级 IPC 合同、单底图缓存、dummy mask、逐层上传、
  分析 `Arc` 共享、原图 `Arc` 复用、流式编码器边界、临时文件发布和 GPU 高水位回收。
- Rust 单元测试：验证 camelCase 序列化、预览/导出边界、60MP 空蒙版计划、`Arc::ptr_eq` 共享、
  60MP 带状缓冲算术、processor 收缩，以及 PNG/TIFF 流式文件的尺寸和 sRGB v4 ICC 往返。
- `npm run synthetic-export:bench`：默认编码 9504×6336 PNG；设置
  `RAW_EDITOR_BENCH_FORMAT=tiff` 可复跑 TIFF。宽高可通过 `RAW_EDITOR_BENCH_WIDTH` 和
  `RAW_EDITOR_BENCH_HEIGHT` 覆盖。

## 合成 60MP 编码基准

2026-08-09 在 Apple M2 Max（12 核）、32 GB、macOS 15.6.1 上运行。测试构造并重复使用与生产
GPU 拼带相同上界的 2048 行 RGBA8 缓冲，再逐行写入真实 PNG/TIFF 编码器和临时文件；RSS 由独立
线程每 10 ms 采样。它不初始化 WGPU、不解码 RAW，也不判断画质，因此不能替代真实相机验收。

| 格式 | 尺寸      |   耗时 |      输出大小 |     峰值 RSS | 生产者带状缓冲 | 旧完整 RGBA8 缓冲 |
| ---- | --------- | -----: | ------------: | -----------: | -------------: | ----------------: |
| PNG  | 9504×6336 | 818 ms |   2,934,463 B | 84,721,664 B |   77,856,768 B |     240,869,376 B |
| TIFF | 9504×6336 | 164 ms | 361,305,558 B | 87,523,328 B |   77,856,768 B |     240,869,376 B |

合成图样高度可压缩，输出大小与耗时不代表真实照片；表中可复用的工程结论是生产者缓冲不随完整
图高增长。驱动对齐、GPU texture 和 RAW 解码内存仍需在端到端基准中单独记录。

## 仍然存在的峰值来源

- 全分辨率导出仍同时需要已解码/变换输入和一张 RGBA16F GPU 输入 texture；本阶段只移除了
  PNG/TIFF 的最终完整 RGBA8 输出图，并未把 RAW 解码或几何变换改造成流式节点。
- JPEG、WebP、JXL、AVIF，以及任何启用 resize/watermark 的导出仍使用完整 CPU 编码图。
  `mozjpeg-rs 0.9.2` 的逐行接口在确定性 60MP 压力图样上触发内部位移溢出，因此 JPEG 暂不进入
  流式选择器；必须先替换或修复编码器并通过同一 harness，不能仅凭小图回归启用。
- PNG 在保留 EXIF 时会暂时把“已压缩输出文件”读回内存交给现有元数据写入器；它不恢复完整
  RGBA8 图，但高熵照片的压缩文件本身仍可能形成明显峰值。TIFF 继续沿用现有“不写 EXIF”限制。
- 高水位回收是导出结束和尺寸切换时的确定性策略，尚未接入操作系统/驱动的实时内存压力通知。
- 活动蒙版本身仍各自占用一张全尺寸灰度 bitmap；本阶段只消除了拼接副本和空蒙版开销。
- 授权真实 60MP RAW 的首屏、滑块延迟、100% ROI、导出耗时和进程峰值仍按计划延期，取得样片后
  必须补做，不得用上述尺寸算术替代。
