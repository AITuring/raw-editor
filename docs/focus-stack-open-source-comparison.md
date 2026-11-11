# 开源图片堆栈与拼接软件：算法调研与对比

> [返回项目进度与任务总览](project-progress.md#stack) · 专项文档。实施状态、优先级和后续任务统一在总览 PF/N9 条目维护，本文不另开任务表。

更新：2026-11-11 联网复核并扩写（上一版为 2026-10 调研，原文完整保存在[归档](archive/focus-stack-open-source-comparison-2026-10.md)，其中更细的缺陷分析、RegistrationProvider 合同草案和真实素材对照清单仍可参考）。本次核对了各项目的公开 README、文档、发布说明和许可证，没有在本机运行外部工具，也没有用我们的素材做跨工具实测。文中外部项目的描述均为转述，速度等数字是项目自述，不是我们的测量。Content was rephrased for compliance with licensing restrictions.

## 0. 结论摘要

1. 开源社区的焦点堆栈工具（focus-stack、ChimpStackr、ShineStacker、PlaneFuse、OpenFocus）针对的都是“固定机位、焦点逐帧变化”的微距/显微堆栈。它们的配准模型以平移/相似/仿射/透视单一全局变换为主，融合算法集中在拉普拉斯金字塔最大值、复小波、深度图、加权平均、导向滤波和学习型源分类。PlaneFuse 文档明确写了不做全景拼接；OpenFocus 只在显微 mosaic 上有少量支持。
2. 拼接工具分成三类：摄影全景（OpenCV Stitcher/OpenStitching、Hugin+Enblend）、显微/切片网格（ASHLAR、MIST、BigStitcher）和学习型视差容忍拼接（UDIS++ 等）。它们普遍把“配准图 + 全局优化 + 接缝/融合”作为固定流程，但都不处理“每个机位还有一组焦点层”。
3. 我们的问题是两者的组合：平面书画、多机位、每机位多焦点层、混合焦段、NEF/ARW RAW、超大画布，并且每个输出像素都要能追溯到来源。开源界目前没有现成项目覆盖这个组合。我们在来源追溯、证据门禁和失败拒绝上明显比开源工具严格；在配准召回、融合方法多样性、速度/GPU 和公开 benchmark 上落后。
4. 最值得借鉴的不是某个新的融合公式，而是几种工程做法：ASHLAR 用置换检验给配准边定阈值，BigStitcher/Hugin 做全局优化并剔除坏边，ChimpStackr/PlaneFuse 提供多融合方法并排对比和 GPU/CPU 自动选择，PlaneFuse 采用 scene-linear RAW 工作域。
5. 许可证是硬约束：我们是 AGPL-3.0。MASt3R（CC BY-NC-SA 4.0）不能集成；StackMFF V3 在 2026-11 改为限制性许可证，v2.0 之后的版本不能直接放进我们的项目；SuperPoint 的权重受限制。LightGlue（不含 SuperPoint）、RoMa、OpenCV、ChimpStackr（GPL-3.0）、ShineStacker（LGPL-3.0）、PlaneFuse/OpenFocus 应用代码（MIT）在许可证层面可以参考或组合，最终仍以法律审查为准。

## 1. 我们当前的管线

当前生产链路（详见[算法说明](focus-stack-algorithm.md)和[架构](focus-stack-architecture.md)）：

```text
RAW preflight/decode（rawler，统一 SourcePreflight）
  -> bounded alignment proxy（约 1,536–2,400px）
  -> FAST/BRIEF（含局部对比度/多尺度变体）+ ratio/scale 匹配
  -> RANSAC：平移 / 相似 / 仿射 / 单应，对称重投影误差
  -> NCC patch refinement + 原生分辨率双向复测
  -> 焦点组 / station 分组
  -> station pose 求解、独立留出、闭环、有界联合场
  -> Virtual_Tile 或 legacy streaming mosaic
  -> 机位内：多标签图割 ownership（清晰度 + 不一致性惩罚）
  -> 机位间：按投影边界距离选 owner，接缝处局部 multiband_blend
  -> 低频 tone、有证据约束的 owner 残差增益
  -> 分块渲染 / RGBA16 TIFF + alpha / Quality Gate / Stack_Report
```

关键实现位置：

- 特征与匹配：`src-tauri/src/panorama_utils/processing/features.rs`（FAST、BRIEF、Hamming、ratio/scale 匹配）
- 图割 owner：`src-tauri/src/panorama_utils/stack_pipeline/focus_fuser/ownership_solver.rs`，有对穷举最优解的性质测试
- 多频带混合：`src-tauri/src/panorama_utils/stitching/filters.rs::multiband_blend`，只用于接缝带（`panorama_seam_band.rs`、`focus_seam_blend.rs`），不是整图融合器
- ONNX 推理基础设施：`ort` 已用于 AI 蒙版、降噪、inpainting，堆栈管线尚未使用

当前有证据的能力范围（摘自[总览](project-progress.md#acceptance)）：阆苑 10 张和文苑 10 张完整原生验收通过；睡觉 9 张 NEF 冷/热基线通过；兰亭 45 张主体色块回归通过，但严格整体质量门禁仍拒绝。84 张、126 张全量尚未验收。9 张 NEF 一次运行的墙钟时间约 330–530 秒，峰值 RSS 约 9–12 GB。

## 2. 焦点堆栈类开源软件

### 2.1 总表

| 项目                                                                        | 许可证                                                                                                  | 配准                                                                                                       | 融合算法                                                                                                                                                     | 工程特征                                                                                                                                          |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| [PetteriAimonen/focus-stack](https://github.com/PetteriAimonen/focus-stack) | MIT                                                                                                     | OpenCV ECC；默认与相邻帧对齐，可改为对参考帧；对齐图默认限制在 2048×2048                                   | Forster 等人的复小波融合；深度图一致性滤波（0–2 级）；小波去噪                                                                                               | C++/OpenCV，OpenCL；`batchsize` 默认 8、最大 32；可只对齐后交给外部工具                                                                           |
| [ChimpStackr](https://github.com/noah-peeters/ChimpStackr)                  | GPL-3.0                                                                                                 | 平移 / 欧氏 / 相似 / 仿射，外加 “breathing” 顺序链式对齐（限制每对帧的尺度步长）；DFT 对齐改编自 imreg_dft | 拉普拉斯金字塔（Wang & Chang 2011）、Hybrid（细频带取最大、粗频带加权，抑制光晕）、加权平均、深度图（多尺度清晰度 + 双边平滑）、Mertens 曝光融合；SML 清晰度 | 全部方法可走 OpenCL GPU；首次运行实测 CPU/GPU 后自动选择；分块（slab）处理超深堆栈；16 位全链路；自述 50 帧 24MP 在 GPU 上低于 11 秒              |
| [ShineStacker](https://github.com/lucalista/shinestacker)                   | LGPL-3.0                                                                                                | SIFT 对齐（可选 silx OpenCL 加速）；亮度/颜色平衡                                                          | PyramidStack、DepthMapStack；`FocusStackBunch` 按帧数加重叠分批                                                                                              | Python API + Qt6 GUI + Jupyter；暗角校正、噪点像素屏蔽；多层 PSD 输出便于手工修图；rawpy 读 RAW                                                   |
| [PlaneFuse](https://github.com/AdrienLF/planefuse)                          | MIT                                                                                                     | 平移 / 相似 / 透视；Lanczos-3 等重采样；亮度匹配；按相关性排除低质量帧的质量门；对齐复核视图               | PMax（金字塔取最大）、DMap（深度图）、Weighted（softmax 加权）、Slab（分层子堆栈）                                                                           | 同相机 scene-linear RAW 流程，输出 Linear DNG（发布前用 LibRaw 重新打开校验，原子发布）；CUDA/MPS，显存不足时自动分块或回退 CPU；明确不做全景拼接 |
| [OpenFocus](https://github.com/Xinzhe99/OpenFocus)                          | 应用 MIT；StackMFF V1/V2 权重另有限制条款                                                               | 有界平移/旋转/尺度搜索 + 采样差分评分，Lanczos4 重采样，裁到有效交集；第三方介绍还提到 ECC 与单应选项      | 十种：OpenFocus A（Burt 金字塔响应）、B（平滑深度图分带合成）、C（金字塔级联）、Guided Filter、DCT、DTCWT、GFG-FGF、StackMFF V1/V2/V3（ONNX）                | 方法并排对比；“来源锁定”修图（只能从贡献帧取像素）；缺失数据图；磁盘后备中间结果；EXIF 焦点排序；v1.77 RC 修复了显微 mosaic 的尺度/旋转问题       |
| [StackMFF V3](https://github.com/Xinzhe99/StackMFF-V3)                      | 2026-11 起改为 StackMFF V3 License（研究和个人使用；进入其他软件或再分发需书面许可）；v2.0 及以前为 MIT | 不负责配准，假设输入已对齐                                                                                 | 逐像素对 N 张输入做分类：默认按概率加权观测值，或硬选概率最高的帧；图像轴无位置编码，因此与输入顺序无关                                                      | 0.156M 参数；公开 benchmark、7,622 个评测输出及 SHA-256 清单                                                                                      |

商业对照（来自 PlaneFuse README 的比较表）：Zerene Stacker 提供 PMax/DMap；Helicon Focus 提供 A/B/C 三种方法，并带 micro-panorama 拼接。开源工具里同时认真处理“堆栈 + 拼接”的几乎没有。

### 2.2 算法要点

**金字塔取最大（PMax / Laplacian pyramid）。** 把每帧分解为拉普拉斯金字塔，每个频带每个位置取对比度最大的系数，最低频带通常取平均或加权，再重建。它对细毛、刚毛这类细节最好，代价是会放大噪声，在高对比边缘附近产生光晕，而且输出值可能不等于任何一帧的观测值。ChimpStackr 的 Hybrid 方法只在细频带取最大，粗频带改为加权，专门用来压光晕。

**复小波（focus-stack）。** 和金字塔思路相同，换成方向选择性更好、近似平移不变的复小波，并对多通道统一决策以减少假颜色。focus-stack 另外加了深度图一致性滤波：低一致性的孤立选择会被邻域覆盖，这能减少噪点，但也可能抹掉纹理背景上的小物体。

**深度图（DMap）。** 先逐像素算清晰度，取最清晰帧的索引作为“深度”，对深度图做平滑（双边或边缘感知），再按深度从源帧取像素。颜色保真最好，因为像素基本直接来自某一帧；缺点是低纹理区域的深度不可靠，前后景交界处容易出现错选或光晕。

**加权平均 / softmax。** 以清晰度为权重做逐像素加权。光晕最少、颜色平滑，但在配准误差下会把两条错位笔画平均成双影，这正是我们在书画场景里拒绝整图平均的原因。

**导向滤波 / GFF、DCT、DTCWT。** OpenFocus 集成的学术方法：基础层/细节层分解后分别融合，或在变换域里按能量选择。它们在公开 MFIF benchmark 上很常见。

**学习型源分类（StackMFF V3）。** 输出是对每个源的概率，而不是直接回归像素值，所以结果不超出观测值范围，也天然给出一个“来源概率图”。这和我们的 owner 思想最接近，但它不处理配准，训练分布是固定视角堆栈，许可证也已收紧。

## 3. 拼接与注册类开源软件

### 3.1 摄影全景

| 项目                                                                                                                              | 许可证     | 核心算法                                                                                                                                             | 与我们相关的点                                                                                                                                                                                                                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [OpenCV Stitcher](https://docs.opencv.org/5.0/main_modules/stitching.html)                                                        | Apache-2.0 | 特征匹配 → 相机旋转估计 / 自标定 → bundle adjustment → warp → seam（图割/动态规划/Voronoi）→ 曝光补偿 → 多频带混合；有 homography 和 affine 两种模型 | 模块可替换；“扫描/平面物体”用 affine 模式，与“全景”分开                                                                                                                                                                                                                          |
| [OpenStitching/stitching](https://github.com/OpenStitching/stitching)                                                             | Apache-2.0 | 基于 OpenCV stitching 模块的 Python 封装，提供 `AffineStitcher`、置信度阈值和 verbose 中间结果（匹配、接缝可视化）；配套最大内接矩形实现             | 中间结果全部落盘、便于定位失败环节的做法值得学；v0.7.0 发布说明提到 OpenCV 5 把 AKAZE/BRISK 移到了 contrib                                                                                                                                                                       |
| [Hugin](https://hugin.sourceforge.io/docs/manual/Hugin.html) / PanoTools                                                          | GPL        | CPFind 控制点 → 镜头/投影模型 → 全局优化 → 曝光/渐晕/白平衡 → nona 重映射 → Enblend/Enfuse                                                           | [mosaic 模式](https://hugin.sourceforge.io/docs/manual/Stitching_a_photo-mosaic.html)用相机平移参数 TrX/TrY/TrZ 描述平面场景下移动的相机，[平面扫描教程](https://hugin.sourceforge.io/tutorials/Mosaic-mode/en.shtml)正对应书画翻拍；2025.0 增加了 enfuse 图形前端 hugin_toolbox |
| [Enblend](https://hugin.sourceforge.io/docs/manual/Enblend.html) / [Enfuse](https://hugin.sourceforge.io/docs/manual/Enfuse.html) | GPL        | Enblend：接缝优化 + 多频带混合；Enfuse：按曝光、对比度、饱和度加权（Mertens 2007）在多分辨率空间融合，也可做景深合成                                 | 可作为我们 owner/graph-cut 路线的经典对照                                                                                                                                                                                                                                        |

### 3.2 显微/切片网格拼接

这类工具面对的是“已知大致位置的规则网格 + 海量瓦片”，和我们的“多机位书画扫描”在结构上最像。

| 项目                                                           | 许可证   | 核心算法                                                                                                                                                                                                                                                                                                                      | 可借鉴点                                                                                                    |
| -------------------------------------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| [ASHLAR](https://github.com/labsyspharm/ashlar)                | MIT      | 用载物台坐标建邻接图 → 每条边相位相关求平移 → 用 NCC 打分 → 从 1,000 对**不相邻**瓦片做置换检验，取 NCC 第 99 百分位作为当前数据集的接受阈值 → 以 −log(NCC) 为权重建最小生成树求位置 → 断开的分量用线性模型外推 → 线性混合 → 金字塔 OME-TIFF（[方法说明](https://labsyspharm.github.io/ashlar/overview/DetCompMethods.html)） | 置换检验能按数据集估计“随机配对能有多像”，直接对应我们书画中重复题字/人物产生假边的问题；比固定阈值更有依据 |
| [MIST](https://github.com/usnistgov/MIST)（NIST）              | 公有领域 | 相位相关 + 载物台模型 + 误差最小化；CPU/GPU 混合，自述处理 TB 级数据比同类快 15–100 倍（[论文](https://isg.nist.gov/deepzoomweb/resources/csmet/pages/image_stitching/downloads/41598_2017_Article_4567.pdf)）                                                                                                                | 用拍摄机构模型约束配准；提供 15 套带参考的精度数据集                                                        |
| [BigStitcher](https://imagej.net/plugins/bigstitcher/)（Fiji） | GPL      | 相位相关检查多个峰 → 按相关系数过滤链接 → 全局优化，可迭代剔除不一致链接 → ICP 精修；与 BigDataViewer 集成，可交互浏览任意大小数据（[全局优化](https://imagej.net/plugins/bigstitcher/global-optimization)）                                                                                                                  | 稀疏连接数据集的全局优化，以及“只加载可见块”的大数据浏览                                                    |

### 3.3 学习型拼接

| 项目                                                        | 许可证 | 核心算法                                                                                                                      | 适用性                                                                     |
| ----------------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| [UDIS++](https://github.com/nie-lang/UDIS2)（ICCV 2023）    | 见仓库 | 无监督，从全局单应过渡到局部薄板样条（TPS）形变；用接缝驱动的合成掩膜代替直接混合（[论文](https://arxiv.org/abs/2302.08207)） | 针对有视差的双图拼接，分辨率低；对平面书画价值有限，TPS 局部形变思路可参考 |
| [Pano360](https://github.com/KiMomota/Pano360)（CVPR 2026） | 见仓库 | 基于 VGGT 类 3D 基础模型，从透视图恢复全景几何，可选 BA                                                                       | 面向 360° 场景，与平面书画无关，只作趋势参考                               |

## 4. 可单独集成的学习组件

| 组件                                                                                    | 许可证                                                                                          | 作用                                                                                                                                   | 对我们的意义                                                                                                            |
| --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| [LightGlue](https://github.com/cvg/LightGlue)                                           | 代码和权重 Apache-2.0；搭配 DISK（Apache）或 ALIKED（BSD-3）可用；**SuperPoint 及其权重受限制** | 稀疏特征匹配网络，自述 RTX 3080 上 1024 点约 150 FPS，CPU 512 点约 20 FPS；有 [ONNX 导出](https://github.com/fabio-sim/LightGlue-ONNX) | 可通过我们已有的 `ort` 运行，作为 FAST/BRIEF 召回不足时的 provider；不能用 SuperPoint 权重                              |
| [RoMa](https://github.com/Parskatt/RoMa) / [RoMaV2](https://github.com/Parskatt/RoMaV2) | RoMa 代码 MIT，DINOv2 Apache-2.0                                                                | 稠密匹配，输出逐像素形变场和置信度；默认 560 → 864 分辨率；另有基于 XFeat 的 Tiny RoMa                                                 | 对跨焦段、大尺度变化、低纹理纸面召回高；稠密置信度可以直接作为 Coverage/支持证据；模型重，适合只在连通分量边界做 rescue |
| [MASt3R](https://github.com/naver/mast3r)                                               | **CC BY-NC-SA 4.0，仅限非商业**                                                                 | 3D 落地的匹配与 SfM                                                                                                                    | 与 AGPL 分发不兼容，只能离线研究对照，不能进入产品                                                                      |
| [StackMFF V3](https://github.com/Xinzhe99/StackMFF-V3)                                  | 2026-11 起限制再分发                                                                            | 已对齐堆栈的学习融合                                                                                                                   | 只能使用 v2.0 及以前的 MIT 副本，或取得书面许可                                                                         |

## 5. 其他意义上的“堆栈”：降噪与 HDR

“图片堆栈”也常指同一画面多帧叠加以降噪或扩展动态范围。它们不解决景深，但 RAW 域处理和异常值剔除可以借鉴。

| 项目                                                                                    | 核心算法                                                                                                                                                                                                                            | 可借鉴点                                                       |
| --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| [Siril](https://siril.readthedocs.io/en/latest/preprocessing/registration.html)（天文） | 配准：星点三角形匹配 + RANSAC、选区互相关、KOMBAT、1.5 版新增行星视频多点配准；叠加：平均 + sigma clipping / winsorized sigma clipping 剔除异常值（[叠加文档](https://siril.readthedocs.io/en/stable/preprocessing/stacking.html)） | 多种配准方法并存、按场景选择；稳健统计剔除                     |
| [HDRMerge](https://github.com/jcelaya/hdrmerge)                                         | 在 RAW 域合并包围曝光，逐像素选噪声最低的来源，输出浮点 DNG 1.4                                                                                                                                                                     | 和 PlaneFuse 一样把结果留在 RAW 开发器可再编辑的域里           |
| [Burst Photo / hdr-plus-swift](https://github.com/martin-marek/hdr-plus-swift)          | 简化版 HDR+：金字塔分块对齐（L2/L1 混合）+ 频域稳健合并（[Hasinoff 2016](http://graphics.stanford.edu/papers/hdrp/hasinoff-hdrplus-sigasia16.pdf)）                                                                                 | macOS 原生 Metal 实现，可参考它在 Apple GPU 上做分块对齐的方式 |

## 6. 算法谱系与我们的位置

### 6.1 配准

| 路线                  | 代表                                     | 优点                                   | 局限                                   | 我们                                              |
| --------------------- | ---------------------------------------- | -------------------------------------- | -------------------------------------- | ------------------------------------------------- |
| 相位相关 / 互相关     | ASHLAR、MIST、BigStitcher、Siril pattern | 快，适合纯平移，亚像素                 | 不处理旋转/尺度/透视；重复纹理有多个峰 | 未作为独立 provider                               |
| ECC / 直接法          | focus-stack、OpenFocus                   | 亚像素精度高，适合小位移的同机位堆栈   | 需要好的初值，大位移会失败             | 有 NCC patch refinement，没有 ECC                 |
| 有界参数搜索          | OpenFocus 自带对齐器                     | 简单稳健                               | 搜索空间有限                           | 无                                                |
| 稀疏特征 + RANSAC     | OpenCV、Hugin CPFind、ShineStacker SIFT  | 通用，处理大位移和透视                 | 低纹理和重复纹理召回差                 | **主路线**：FAST/BRIEF + RANSAC，模型从平移到单应 |
| 学习型稀疏 / 稠密匹配 | LightGlue、RoMa                          | 跨尺度、低纹理召回高，稠密方法给置信度 | 重、需要 GPU，训练分布不同             | 未接入                                            |

| 全局求解                      | 代表                        | 我们                                                  |
| ----------------------------- | --------------------------- | ----------------------------------------------------- |
| 最小生成树，不处理闭环        | ASHLAR                      | 不采用                                                |
| 全局优化 + 剔除坏链接         | BigStitcher、MIST           | 部分具备：closure、独立留出                           |
| bundle adjustment（相机模型） | OpenCV、Hugin               | 没有完整的稀疏 BA；用 station solver + 有界联合场代替 |
| 平面 mosaic 平移模型          | Hugin mosaic（TrX/TrY/TrZ） | 二维机位图 + 平面单应，思路相近                       |

### 6.2 融合

| 路线                            | 代表                                     | 像素来自观测             | 光晕/双影风险          | 我们                              |
| ------------------------------- | ---------------------------------------- | ------------------------ | ---------------------- | --------------------------------- |
| 金字塔取最大                    | ChimpStackr、PlaneFuse PMax、Zerene PMax | 否                       | 光晕、噪声放大         | 无整图实现                        |
| 复小波 / DTCWT                  | focus-stack、OpenFocus                   | 否                       | 较低                   | 无                                |
| 深度图                          | ChimpStackr、PlaneFuse DMap、OpenFocus B | 基本是                   | 交界处错选             | 机位内 owner 本质上是深度图的一种 |
| 加权 / softmax                  | ChimpStackr、PlaneFuse                   | 否                       | 错位时双影             | 主动拒绝用于细节                  |
| 导向滤波 / GFF                  | OpenFocus                                | 否                       | 中                     | 无                                |
| 学习型源分类                    | StackMFF V3                              | 是（硬选）/ 近似（加权） | 依赖训练分布           | 无                                |
| **图割硬 owner + 接缝带多频带** | 我们；OpenCV 的 seam finder 思路相近     | **是，可逐像素追溯**     | 接缝可见、硬切换处断笔 | **主路线**                        |

## 7. 我们与开源方案的优劣

### 7.1 我们的优势

1. **问题覆盖面。** 多机位 × 每机位多焦点层 × 混合焦段 × 平面书画 × RAW × 超大画布，开源工具里没有同时覆盖这些的。焦点堆栈工具不做拼接，拼接工具不做逐机位景深合成。
2. **来源追溯。** 输出像素有 owner、Coverage、alpha 和 Stack_Report，可以从最终画布回放到 Source_RAW。最接近的开源做法是 OpenFocus 的“来源锁定修图”和缺失数据图、StackMFF V3 的源概率，但它们没有贯穿配准和输出的证据链。
3. **证据门禁与拒绝。** 没有可验证重叠就拒绝输出，有独立留出、四区分布、尺度重拟合和闭环 P95 门槛。PlaneFuse 有按相关性排除帧的质量门，ASHLAR 有置换检验阈值，但都没有这种多层验收。
4. **不平均错位细节。** 硬 owner 避免了加权/金字塔方法在配准残差下产生的双影，这对书法笔画很关键。
5. **可复现性。** 冻结二进制、输入/输出 SHA-256、冷/热回归和 TIFF 摘要比对，严格程度超过开源工具的常见做法（OpenFocus 和 PlaneFuse 也有 CI 和冒烟测试，但没有真实素材的逐像素回归）。
6. **原生输出合同。** RGBA16 TIFF、未关联 alpha、ICC、原子发布，有逐像素独立审计。

### 7.2 我们的劣势

1. **速度和资源。** 9 张 NEF 一次运行需 330–530 秒、9–12 GB RSS。ChimpStackr 自述 50 帧 24MP 在 GPU 上不到 11 秒。两者任务不同（我们包含跨机位几何、原生复测和 QG），不能直接对比，但差距是数量级的，而且我们的堆栈管线还没有 GPU 路径。
2. **配准召回。** FAST/BRIEF 对低纹理纸面、重复题字和跨焦段尺度变化召回不足。阆苑 0–2 的 1,061 个独立探针中只有 23 个通过（v38–v41 记录），最终靠局部双补丁复测才过门槛。开源界已经普遍可选 SIFT、ECC，学习型匹配器也成熟可用。
3. **阈值来源。** 大量固定阈值（3px、1%、相关系数 0.8/0.9 等）是按素材逐步校准的。ASHLAR 用置换检验从数据本身估计阈值，更有统计依据，也更容易迁移到新素材。
4. **融合方法单一。** 只有一条 owner/graph-cut 路线，没有 PMax、小波、深度图、Mertens 等可并排比较的基线。ChimpStackr、PlaneFuse、OpenFocus 都把“方法对比”做成了一等功能，便于判断问题出在配准还是融合。
5. **接缝与色调。** 硬 owner 的代价是曝光/反光差异下接缝可见。兰亭 v54–v60 多轮 tone 调整说明这是持续成本。Enblend 的整图多频带混合和 Hugin 的曝光/渐晕/白平衡联合优化在这方面更成熟；我们的 `multiband_blend` 只用于接缝带。
6. **RAW 工作域。** 我们的 RAW 路径包含面向显示的处理；PlaneFuse 在 scene-linear 相机 RGB 中融合并输出 Linear DNG，HDRMerge 输出浮点 DNG，两者都把调色留给 RAW 开发器。
7. **外部基准。** 没有公开 MFIF benchmark 的 PSNR/SSIM、QAB/F 对照，也没有在相同输入上跑过 Hugin、OpenCV 或 focus-stack 做交叉验证。
8. **交互修复。** 开源工具普遍提供从源帧手工修图（ShineStacker 多层输出、PlaneFuse/OpenFocus 来源锁定修图）。我们目前没有面向堆栈结果的修图入口。

### 7.3 能力差距矩阵

| 能力            | 当前状态                                 | 开源最佳实践                           | 差距     | 对应任务    |
| --------------- | ---------------------------------------- | -------------------------------------- | -------- | ----------- |
| RAW 读取        | rawler 生产 loader，统一 SourcePreflight | scene-linear + Linear DNG（PlaneFuse） | 中       | PF.11       |
| 低分辨率分析    | bounded proxy，运行内 source 缓存        | 跨运行 proxy/feature 缓存              | 中       | PF.7        |
| 特征            | FAST/BRIEF 及变体                        | SIFT / ECC / LightGlue / RoMa 可选     | 高       | PF.10       |
| 候选搜索        | ≤64 张全成对，更大用有界候选             | 邻接图 + 检索 + 边界 rescue            | 高       | PF.8        |
| 边接受阈值      | 固定阈值                                 | 置换检验（ASHLAR）                     | 中高     | PF.8 / N4.5 |
| 全局几何        | station solver、闭环、有界联合场         | BA、迭代剔除坏链接                     | 中       | N4.5 / N9.1 |
| 融合            | 图割 owner + 接缝带多频带                | 4–10 种方法并排                        | 高       | PF.10       |
| 接缝/色调       | 低频 tone + 证据约束 owner 增益          | 整图多频带 + 曝光/渐晕联合优化         | 中高     | N6.2 / N6.3 |
| GPU             | 堆栈管线无                               | OpenCL/CUDA/MPS + CPU 回退             | 高       | PF.11       |
| 诊断            | Stack_Report、QG、诊断导出               | 中间结果落盘、对齐复核视图             | 中       | N3.3        |
| 外部基准        | 无                                       | MFIF benchmark、MIST 参考数据集        | 中高     | N9.2        |
| 来源追溯 / 门禁 | 强                                       | 弱                                     | 我们领先 | —           |

## 8. 许可证与可集成性

项目许可证为 AGPL-3.0-only（`LICENSE`、`package.json`、`src-tauri/Cargo.toml`）。

| 来源                                                        | 许可证          | 结论（非法律意见）                                            |
| ----------------------------------------------------------- | --------------- | ------------------------------------------------------------- |
| OpenCV、OpenStitching、LightGlue（不含 SuperPoint）、DINOv2 | Apache-2.0      | 可组合                                                        |
| RoMa 代码、PlaneFuse、OpenFocus 应用、focus-stack、ASHLAR   | MIT             | 可组合，保留版权声明                                          |
| ChimpStackr、Hugin、Enblend/Enfuse、BigStitcher             | GPL             | GPLv3 第 13 条允许与 AGPLv3 组合；GPLv2-only 的部分需单独确认 |
| ShineStacker                                                | LGPL-3.0        | 可组合                                                        |
| MIST                                                        | 美国政府作品    | 通常可用，需确认仓库声明                                      |
| SuperPoint 权重                                             | 限制性          | 不可用                                                        |
| MASt3R / DUSt3R                                             | CC BY-NC-SA 4.0 | 不可集成，只能离线研究                                        |
| StackMFF V3（2026-11 之后）、StackMFF V1/V2 重训权重        | 限制再分发      | 不可直接集成；只能用 v2.0 及以前的 MIT 副本                   |

实际集成更现实的方式是：参考算法自行用 Rust 实现，或通过 `ort` 加载许可证允许的 ONNX 权重。不建议在生产路径中引入 Python 运行时。

## 9. 上一版建议的执行状态

上一版（2026-10）提出的 P0–P3，现状如下（以[总览](project-progress.md#stack)为准）：

| 上一版建议                                 | 当前状态                                                            |
| ------------------------------------------ | ------------------------------------------------------------------- |
| P0：singleton scan 与真正多帧 station 分开 | 完成（PF.1）                                                        |
| P0：SourcePreflight 合并读取               | 完成（PF.2）                                                        |
| P0：阶段计时、RSS、缓存计数                | 完成（PF.1、PF.6 八轮交错归因）                                     |
| P0：诊断写盘默认关闭                       | 诊断只在显式目录下启用                                              |
| render/QG 来源复用、缓存预算               | 完成（PF.3、PF.4）                                                  |
| P1：可替换注册器 RegistrationProvider      | 未开始（PF.10）                                                     |
| P1：match graph / coarse-to-fine           | 未开始（PF.8）                                                      |
| P1/P2：经典融合基线                        | 未开始（PF.10）                                                     |
| P2：scene-linear RAW                       | 未开始（PF.11）                                                     |
| P2：持久化缓存与 tiled pipeline            | 部分：运行内缓存完成，跨运行和按需 tile 未完成（PF.7、N6.4）        |
| P3：learned fusion、GPU                    | 未开始（PF.11）；StackMFF V3 许可证变化后需改用 v2.0 副本或其他模型 |

按总览的顺序，这些都排在兰亭 N6.5 收口和两套 10 张稳定复验之后。

## 10. 建议

以下建议都映射到现有任务编号，不新开任务。每项都应先做对照实验，有可复现收益后再考虑改默认行为。

### 10.1 置换检验阈值（PF.8 / N4.5，成本低）

借鉴 ASHLAR：在同一素材中随机抽取确定不相邻的机位对（例如用文件夹或已知拍摄网格排除相邻对），跑同样的匹配和 NCC 评分，取高分位数作为“随机也能达到的相似度”。只把它作为附加证据写进 Stack_Report，先观察它与现有固定门槛的一致程度，不替换门槛。这能直接检验重复题字/人物产生的假边。

### 10.2 注册 provider 对照（PF.10）

固定同一组 proxy，比较：

1. 当前 FAST/BRIEF；
2. SIFT 或 AKAZE（OpenCV contrib，或纯 Rust 实现）；
3. 当前特征 + ECC 局部精修；
4. LightGlue + DISK/ALIKED（ONNX，经 `ort`）；
5. RoMa，只用在连通分量边界的 rescue。

记录候选召回、通过验证的边、连通分量数、留出 P95、耗时和峰值 RSS。provider 只提交候选和证据，最终几何门禁仍由现有 Rust 路径决定。阆苑 0–2 是最直接的测试边。

### 10.3 融合基线并排（PF.10）

在同一组已配准的机位内帧上运行：当前 owner/graph-cut、PMax、Hybrid、DMap、DTCWT、Mertens 式加权、OwnerHybrid（笔画和印章用硬 owner，纸面低频用多频带）。每种方法都输出源索引或置信度，统计 ghost edge、光晕率、跨高梯度接缝数、tone_step_p95 和 provenance 完整度。ChimpStackr（GPL-3.0）和 focus-stack（MIT）的实现可作为离线参考结果。

### 10.4 外部工具交叉验证（N9.2）

- 用 Hugin mosaic 模式（只优化 TrX/TrY/Roll 等）在文苑或阆苑 10 张上求机位，与我们的 station pose 比较；
- 用 OpenStitching `AffineStitcher` 的 verbose 输出对照匹配和接缝；
- 用 focus-stack / ChimpStackr 在单个机位的焦点层上生成参考，与我们机位内 owner 结果做局部比较。

外部结果只作为诊断参考，不作为生产像素或关系证据，与现有 PS 参考的处理规则一致。

### 10.5 GPU 与 scene-linear（PF.11）

两点值得参考：ChimpStackr 的“实测 CPU/GPU 后自动选择”，以及 PlaneFuse 的“显存不足时分块或回退 CPU”。实现前先用 PF.6 的 profile 确认瓶颈；station solve、来源证据和 QG 保持 CPU 可审计路径。scene-linear 方面，在线性域做配准和融合，显示变换只在预览和导出时应用，并记录 RAW 设置和镜头校正版本。

### 10.6 来源锁定修图（N5.5）

ShineStacker、PlaneFuse 和 OpenFocus 都提供“只从贡献帧取像素”的修图功能，这与我们的 owner 模型天然契合：用户改的是 owner，而不是画像素。2026-11-11 用户确认立项，任务范围和进入条件见[总览 N5.5](project-progress.md#n5)。

## 11. 结论

开源社区在单项技术上已经很成熟：焦点融合有多种公开算法和带 GPU 的桌面产品，拼接有 OpenCV/Hugin 的完整流程和显微领域的大规模网格方案，学习型匹配器的许可证也逐渐宽松。但没有一个项目同时解决多机位、多焦点层、平面书画、RAW 和来源可追溯这个组合，这正是我们的定位。

我们的主要短板集中在三处：配准召回（特征单一，阈值固定）、融合方法单一（无可比较基线）、速度（无 GPU，单次运行数分钟到十分钟级）。建议先做成本最低的置换检验阈值和注册 provider 对照，用阆苑 0–2 这类已知难边验证收益；融合基线和 GPU 放在其后。所有这些都排在兰亭 N6.5 收口之后。

## 12. 参考

焦点堆栈

- [PetteriAimonen/focus-stack](https://github.com/PetteriAimonen/focus-stack) · [参数文档](https://github.com/PetteriAimonen/focus-stack/blob/master/docs/focus-stack.md)
- [ChimpStackr](https://github.com/noah-peeters/ChimpStackr)
- [ShineStacker](https://github.com/lucalista/shinestacker) · [文档](https://github.com/lucalista/shinestacker/blob/main/docs/main.md)
- [PlaneFuse](https://github.com/AdrienLF/planefuse/blob/main/README.md)
- [OpenFocus](https://github.com/Xinzhe99/OpenFocus) · [融合方法](https://github.com/Xinzhe99/OpenFocus/blob/main/docs/FUSION_METHODS.md)
- [StackMFF V3](https://github.com/Xinzhe99/StackMFF-V3)

拼接与注册

- [OpenCV Images Stitching](https://docs.opencv.org/5.0/main_modules/stitching.html) · [详细示例](https://docs.opencv.org/4.13.0/d8/d19/tutorial_stitcher.html)
- [OpenStitching/stitching](https://github.com/OpenStitching/stitching) · [发布说明](https://github.com/OpenStitching/stitching/releases)
- [Hugin](https://hugin.sourceforge.io/docs/manual/Hugin.html) · [2025.0.0 发布说明](https://hugin.sourceforge.io/releases/2025.0.0/en.shtml) · [Photo-mosaic](https://hugin.sourceforge.io/docs/manual/Stitching_a_photo-mosaic.html) · [平面扫描教程](https://hugin.sourceforge.io/tutorials/Mosaic-mode/en.shtml)
- [Enblend](https://hugin.sourceforge.io/docs/manual/Enblend.html) · [Enfuse](https://hugin.sourceforge.io/docs/manual/Enfuse.html)
- [ASHLAR](https://github.com/labsyspharm/ashlar) · [计算方法](https://labsyspharm.github.io/ashlar/overview/DetCompMethods.html)
- [MIST](https://github.com/usnistgov/MIST) · [论文](https://isg.nist.gov/deepzoomweb/resources/csmet/pages/image_stitching/downloads/41598_2017_Article_4567.pdf)
- [BigStitcher](https://imagej.net/plugins/bigstitcher/) · [全局优化](https://imagej.net/plugins/bigstitcher/global-optimization)
- [UDIS++](https://github.com/nie-lang/UDIS2) · [论文](https://arxiv.org/abs/2302.08207)
- [Pano360](https://github.com/KiMomota/Pano360)

学习组件

- [LightGlue](https://github.com/cvg/LightGlue) · [LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX)
- [RoMa](https://github.com/Parskatt/RoMa) · [RoMaV2](https://github.com/Parskatt/RoMaV2)
- [MASt3R](https://github.com/naver/mast3r)

降噪与 HDR

- [Siril 配准](https://siril.readthedocs.io/en/latest/preprocessing/registration.html) · [Siril 叠加](https://siril.readthedocs.io/en/stable/preprocessing/stacking.html)
- [HDRMerge](https://github.com/jcelaya/hdrmerge)
- [hdr-plus-swift / Burst Photo](https://github.com/martin-marek/hdr-plus-swift) · [Hasinoff et al. 2016](http://graphics.stanford.edu/papers/hdrp/hasinoff-hdrplus-sigasia16.pdf)

论文

- [Brown and Lowe, Automatic Panoramic Image Stitching using Invariant Features](https://mattabrown.github.io/pdf/ijcv2007.pdf)
- [Forster et al., Complex Wavelets for Extended Depth-of-Field](https://bigwww.epfl.ch/preprints/forster0404p.pdf)
- [Mertens et al., Exposure Fusion](https://jankautz.com/publications/exposure_fusion.pdf)
- [Multi-focus Image Fusion: A Survey of the State of the Art](https://www.sciencedirect.com/science/article/pii/S1566253520303109)
