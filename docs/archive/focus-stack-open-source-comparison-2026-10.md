# 开源堆栈项目与成熟算法对比（2026-10 版，归档）

> 归档说明：本文是 2026-11-11 扩写前的原版正文，只追溯，不派发任务。当前版本见[开源图片堆栈与拼接软件调研](../focus-stack-open-source-comparison.md)。原文件未纳入 git，此处按扩写前读取的内容原样保存，相对链接已改为指向上级目录。

本文把当前 `raw-editor` 的焦点堆栈/书画拼接管线，与公开的开源项目和成熟算法路线进行对照。目标不是照搬某个项目，而是区分：

1. 哪些能力已经达到工程可用水平；
2. 哪些能力属于当前明显短板；
3. 哪些改进可以直接落地；
4. 哪些方向需要数据集、模型或较大架构投入。

## 1. 当前管线的真实定位

当前实现不是传统“固定相机、同尺寸、同位置”的普通 focus stack，而是一个混合问题：

- 一部分输入是同一机位的多焦点图层；
- 一部分输入是不同机位的移动扫描或书画拼图；
- 不同机位可能有横向/纵向位移、轻微旋转、尺度变化和焦段变化；
- 输入主要是 NEF/RAW，需要生产 RAW loader、镜头校正、曝光归一化和高分辨率输出；
- 结果必须保留 source owner、coverage、confidence 和 provenance，不能用生成纹理掩盖缺失来源。

当前生产链路大致为：

```text
RAW preflight/decode
  -> bounded alignment proxy
  -> FAST/BRIEF features and pair matching
  -> RANSAC transform verification
  -> focus/capture grouping
  -> station pose/closure solve
  -> Virtual_Tile or legacy mosaic path
  -> ownership/graph-cut focus fusion
  -> low-frequency tone and residual correction
  -> tiled render/export and Quality Gate
```

这条链路的工程化程度已经高于普通脚本：有取消和 RSS 监控、来源校验、Quality Gate、Virtual_Tile 数据结构、报告和真实素材验收。不过它同时承担了“全景注册器、书画平面拼接器、焦点融合器、RAW 开发器和大画布渲染器”五种职责，模式边界和通用注册能力因此成为主要风险。

## 2. 值得对照的开源项目

| 项目 | 核心路线 | 已经成熟的部分 | 对当前项目的直接启发 |
| --- | --- | --- | --- |
| [OpenCV Stitcher](https://docs.opencv.org/5.0/main_modules/stitching.html) | 特征匹配、相机/仿射模型、bundle adjustment、warp、seam、exposure compensation、blender | 模块边界清晰，同一套 pipeline 可替换 matcher、模型、seam 和 blender；同时支持 homography 和 affine | 将“书画扫描”和“摄影全景”明确拆成不同注册模型；补齐可替换的 matcher、模型和 blender 基线 |
| [Hugin / PanoTools](https://hugin.sourceforge.io/docs/manual/Hugin.html) | 控制点、镜头/投影模型、全局优化、曝光/渐晕/白平衡、mask、Enblend/Enfuse | 面向多行、多镜头、扫描图和 bracket 的完整工作流，支持手工复核和批处理 | 建立可审计的 match graph、控制点复核和镜头模型；不要只依赖一个自动路径 |
| [Enblend](https://hugin.sourceforge.io/docs/manual/Enblend.html) / [Enfuse](https://hugin.sourceforge.io/docs/manual/Enfuse.html) | 多频带接缝融合；基于曝光、对比度、饱和度的多分辨率 exposure/focus fusion | 算法简单、可解释、适合大图；能把低频色调和高频细节分开处理 | 当前单一 owner/graph-cut 路径需要一个可对照的经典多频带融合 baseline |
| [PetteriAimonen/focus-stack](https://github.com/PetteriAimonen/focus-stack) | OpenCV `findTransformECC` 对齐，复杂小波融合，深度图，可选全分辨率对齐 | 默认最大约 2048px 做对齐、支持邻居/全局对齐、batchsize、线程和 OpenCL；明确暴露内存/速度参数 | 当前可采用“低分辨率配准 + 原生分块融合 + bounded batch”的架构，并建立可重复的资源预算 |
| [PlaneFuse](https://github.com/AdrienLF/planefuse) | PMax、DMap、weighted、Slab 四种融合；线性 RAW；CUDA/MPS；自动 tiled/CPU fallback | 把对齐、融合方法比较、RAW 线性工作流、GPU 和 OOM fallback 放到同一产品协议 | 增加多方法对照和线性 RAW 模式；GPU 之前先实现 CPU tiled fallback 与方法级报告 |
| [OpenFocus](https://github.com/Xinzhe99/OpenFocus) | 多种注册/融合模式、CLI、批处理、深度图、对比和来源锁定 retouch | 让用户可以并排比较算法，而不是把单一算法当作唯一答案 | 当前产品需要“算法候选 + alignment review + 结果对比”层，方便定位是配准错还是融合错 |
| [StackMFF V3](https://github.com/Xinzhe99/StackMFF-V3) | 对每个像素分类其最可能的输入源，支持任意数量和任意顺序，概率加权或 hard selection | 变量长度、顺序不敏感、输出值来自观测源；提供 benchmark、权重和评估工具 | 可作为“已注册 stack 的可选融合器”，但不能直接替代当前几何、provenance 和书画 shifted-mosaic 约束 |

这些项目解决的问题并不完全相同：OpenCV/Hugin 更偏注册和全景，Enblend/Enfuse 更偏融合，focus-stack/PlaneFuse/OpenFocus 更偏固定视角焦点堆栈。当前项目的核心任务是把它们的成熟部件组合成明确的模式，而不是寻找一个能覆盖所有输入的单一算法。

## 3. 业界和论文中的成熟路线

### 3.1 全景注册：特征图、几何验证和全局优化

Brown–Lowe 的经典全自动全景路线使用具有尺度/旋转/光照鲁棒性的局部特征，在无序图片中构建匹配图，通过 RANSAC 验证关系，再用 bundle adjustment 联合优化相机参数，最后做增益补偿和多频带融合。[论文](https://mattabrown.github.io/pdf/ijcv2007.pdf)

OpenCV 的公开 pipeline 也将这些阶段拆开：features/matching、rotation estimation、autocalibration、warping、seam estimation、exposure compensation 和 blending，并明确区分 homography 模型与 affine 模型。[官方文档](https://docs.opencv.org/5.0/main_modules/stitching.html)

### 3.2 焦点融合：多尺度、深度图、质量权重和一致性

经典 focus stack 主要有四类：空间域 focus measure；Laplacian pyramid、DWT、complex wavelet 等变换域方法；焦点位置/深度图方法；学习型来源概率方法。

Forster 等人的 complex-wavelet 方法强调多通道扩展和抑制伪影/假颜色。[论文](https://bigwww.epfl.ch/preprints/forster0404p.pdf)

Enfuse 采用 Mertens–Kautz–Van Reeth exposure fusion，把曝光、对比度和饱和度作为权重，在多分辨率空间融合，避免先生成 HDR；该路线同样可以用于 extended DOF。[算法说明](https://hugin.sourceforge.io/docs/manual/Enfuse.html)

StackMFF V3 则把融合建模为“每个像素选择哪个源”，输出可以是概率加权或 hard selection，并强调输入顺序不应影响结果。[项目与方法说明](https://github.com/Xinzhe99/StackMFF-V3)

### 3.3 工程化性能：代理图、batch、tile、GPU fallback

成熟工具普遍不会对所有源图都做完整分辨率、完整精度、完整 pair 搜索：focus-stack 默认把对齐限制在约 2048px，并提供 `batchsize`、线程数和 OpenCL 选项；PlaneFuse 使用 GPU 时仍保留 tiled/CPU fallback，并报告时间和内存；Hugin 把匹配、优化和渲染拆成可独立执行的阶段，支持后台批处理。

当前项目已经有 bounded proxy、RSS guard 和 streaming compositor，但还没有把这些能力统一成一个可复用的 run-level resource contract。

## 4. 当前算法的主要缺陷

### 4.1 注册器仍然偏“自定义单一路线”

当前核心特征路径是 FAST/BRIEF，bounded 场景才使用多尺度 proxy；这对普通纹理和小位移有效，但面对书画的重复题字、低纹理纸面、跨焦段和较大尺度变化时，召回率不足。

与成熟路线相比，主要缺少：

- SIFT/AKAZE/KAZE/更强二进制描述子或可选 learned descriptor；
- 基于局部图像的 ECC/DIS/相关性 refinement；
- 对匹配图的概率化验证，而不只是固定 inlier/scale/support 门；
- 一个成熟的 sparse bundle adjustment 层，联合优化所有相机/机位参数和闭环关系。

当前已有 RANSAC、station solver、closure 和多种 rescue，但这些组件仍由自定义阈值和回退顺序驱动。历史 84 张素材中曾出现大量 pair 未测量、少量关系通过和多个连通分量，这说明主要瓶颈首先是召回和模型覆盖，而不是最后一层阈值不够宽。

### 4.2 几何模式边界不够清晰

当前同时处理固定机位 focus stack、shifted mosaic、平面书画扫描、普通照片 panorama 和 mixed focal length。这些输入不应只由一个自动入口通过 heuristics 选择。OpenCV 已明确把照片 panorama 的 homography 模型与扫描/物体的 affine 模型分开；Hugin 也把镜头、投影、控制点和 mask 作为可审查的独立层。

当前最明显的实际缺陷是：移动扫描可能被误判为多帧 station，从而进入 layered Virtual_Tile 和 graph-cut 路径，导致耗时和 RSS 远高于 legacy streaming 路径。这个问题不是“再加几个线程”能解决的，而是路径选择和模式分类问题。

### 4.3 Pair 搜索和 matcher 成本过高

当前小堆栈仍可能 exhaustive pair search；同时 pair 内的描述子匹配会产生较大的临时距离矩阵，外层 pair、RANSAC 和内部 Rayon 可能互相过度订阅。

成熟方案通常采用低成本候选检索、top-K 邻接和视觉候选、必要时再做完整几何验证，图不连通时只对 component boundary 做 bounded rescue。当前大堆栈已有部分视觉检索，但小堆栈和 mixed-focal 分支仍有重复计算，且多种 ratio/scale 候选可能重复建立距离数据。

### 4.4 融合方法过于集中在单一 owner/graph-cut 路线

当前的 Ownership、Sharpness_Confidence、Coverage、graph cut 和低频 tone correction 对来源追溯很强，但会带来几个问题：

- 高梯度处硬 owner 切换可能产生接缝、断笔或局部不连续；
- 低纹理区域如果只依赖 sharpness，容易把曝光/色调差异误认为内容质量；
- 多帧焦点层、不同机位和大画布之间没有统一的多尺度融合 baseline；
- 目前缺少 PMax、DMap、weighted、wavelet、Mertens-like fusion 的并排结果，难以判断问题来自注册还是融合。

成熟工具通常保留多种融合器：Enfuse 的质量权重和多分辨率融合、focus-stack 的复杂小波、PlaneFuse 的 PMax/DMap/weighted/Slab，以及 OpenFocus 的多算法 side-by-side。当前应该把这些方法作为可比较的候选，而不是一次性替换现有 owner 路径。

### 4.5 RAW 颜色和几何基准还不够“场景线性”

当前生产 RAW 路径包含 demosaic、默认 RAW enhancement、曝光显示归一化、镜头校正和后续 tone 处理。这样做有利于直接得到可视化图像，但会把“为了显示的处理”和“为了配准/融合的测量”混在一起。

PlaneFuse 的公开 RAW 方案强调 same-camera、scene-linear、无白平衡/伽马/降噪/锐化烘焙，并输出可重新进入 RAW developer 的 Linear DNG。它的同相机限制不适合直接照搬到当前混合焦段书画场景，但说明当前应该增加一个可审计的 scene-linear working domain：配准使用稳定的线性 luma/结构代理，融合保留原始线性 RGB 比例，显示预览和最终导出再单独应用 display transform，并记录 RAW 设置和 lens calibration 版本。

### 4.6 性能架构还有明显重复工作

当前实现仍存在：

- preflight、digest、EXIF、exposure 分多次读源文件；
- scalable 路径在 render/Quality Gate 重新解码；
- `VirtualTileStore` 已实现但生产命中链路不完整；
- pair candidate、matcher 和 RANSAC 的重复计算；
- RGB32F 整帧 clone；
- nested Rayon 和并发 RAW decode 造成 RSS 抖动；
- 诊断 JSON 写盘可能混入性能基线。

成熟开源工具通常把 proxy、batch、tile、线程和 GPU fallback 作为一等配置，且明确报告时间/内存。当前需要把这些能力从零散 helper 提升为 run-level resource contract。

### 4.7 验证体系还缺少外部标准对照

当前真实素材验收在来源、覆盖、接缝和 Quality Gate 上较严格，这是优点；但还缺少公开多焦点 benchmark 的 PSNR/SSIM、QAB/F、MEF-SSIM 对照，传统方法、当前方法和学习方法的固定 ablation，source-order shuffle/冗余帧/缺帧/不同焦段的统一测试表，以及 per-region artifact rate、halo rate、owner stability。

MFIF survey 和 MFIF library 公开了代表性算法、数据集和评价指标，StackMFF V3 也公开了 benchmark、训练/评估工具和归档输出。当前可以借用其评价框架，但不能把普通固定视角 benchmark 的分数直接当作书画 shifted-mosaic 的验收结论。

## 5. 当前能力与成熟路线的差距矩阵

| 能力 | 当前状态 | 成熟路线 | 差距等级 | 建议 |
| --- | --- | --- | --- | --- |
| RAW 读取 | 生产 NEF loader，支持 lens/exposure | scene-linear RAW、可复现处理设置、Linear DNG | 中 | 增加线性工作域和设置 hash，保留现有显示模式 |
| 低分辨率分析 | 已有 1,536–2,400px bounded proxy | proxy 与 full render/cache 完全分层 | 中 | 先统一 SourcePreflight、proxy cache 和 full-res lazy decode |
| 特征 | FAST/BRIEF，多尺度只在部分场景使用 | SIFT/AKAZE/ECC/learned descriptor 可选 | 高 | 做 descriptor provider，不直接删除 FAST/BRIEF |
| 候选搜索 | 邻接、自然排序、部分 retrieval | match graph、概率验证、bounded rescue | 高 | coarse retrieval + top-K + component boundary rescue |
| 几何 | RANSAC、homography/affine、station solver、closure | camera model + sparse BA + global robust optimization | 中高 | 先做明确模式分类，再补 station-aware BA |
| 融合 | owner/coverage/confidence/graph cut/tone | multiband、wavelet、quality-weighted、depth-map、AI | 高 | 先增加可比较 classical baselines，再做 hybrid owner/soft blend |
| 大画布 | streaming/tile 和 RSS guard 已有 | batch/tile/GPU fallback 与稳定 cache | 中 | 接入 Virtual_TileStore 和 render/Quality Gate LRU |
| 诊断 | Stack_Report、Quality Gate、provenance | alignment review、method comparison、公开 benchmark | 中高 | 增加匹配图、owner map、depth/confidence 和方法对照输出 |
| 性能 | 部分并行，但有重复 decode、距离矩阵和嵌套并行 | 分阶段预算、batch、GPU/tiled fallback | 高 | 按性能优化文档的阶段 0–6 落地 |

## 6. 未来提升路线

### 6.1 P0：先把模式分类和性能基础设施做对

1. 把 singleton scan 与真正的多帧 focus station 分开，避免移动扫描误走 layered path。
2. 建立 `SourcePreflight/RunSourceContext`，合并尺寸、EXIF、曝光、digest 和 RAW metadata 读取。
3. 为 prepare、match、group solve、tile fuse、render、Quality Gate 写完整 stage timing、RSS 和 cache counters。
4. 让诊断写盘默认关闭，不把诊断 IO 混入性能基线。

这一步不改变融合算法，但很可能带来最大的实际收益。

### 6.2 P1：建立可替换的注册器

建议定义统一的 `RegistrationProvider` 合同：

```text
input: source proxy + metadata + mode hint
output: candidate correspondences + transform hypotheses + confidence
required: coordinate frame, source digest, provider version, diagnostics
```

至少提供三条 provider：

1. `FastBriefProvider`：当前 FAST/BRIEF，适合低成本默认路径；
2. `ScaleRobustProvider`：SIFT/AKAZE/KAZE 或等价尺度鲁棒特征；
3. `LocalRefineProvider`：ECC/DIS/NCC，在已有粗 homography 后做局部精修。

provider 只能提交候选和证据，最终 geometry gate、Station_Relation 和 closure 仍由 Rust 主路径决定。

### 6.3 P1：从 exhaustive pair 转为 match graph

先做紧凑描述子检索，再对 top-K 和邻接 pair 做完整验证。图不连通时只对 component boundary 做 rescue，并在报告中记录：candidate edge、coarse pass edge、verified edge、reject reason、component count、rescue budget 和 rescue 命中数。

这样既能提升速度，也能直接定位“没有关系”到底是没有候选、描述子不相似、RANSAC 失败还是 Station_Relation 门禁拒绝。

### 6.4 P1/P2：增加经典融合基线

在不破坏 provenance 的前提下，增加以下可选方法：

- `FocusMax`：局部 Laplacian/gradient 最大响应；
- `FocusWavelet`：complex wavelet 或 DWT 多尺度融合；
- `FocusDMap`：focus measure → depth/confidence map → edge-aware smoothing；
- `FocusWeighted`：softmax/质量权重融合；
- `ExposureFocusFusion`：借鉴 Mertens 的 contrast/exposure/saturation 权重；
- `OwnerHybrid`：高梯度和文字/印章区域使用 source owner，低频纸面使用 soft multiband。

所有方法都需要输出 source index/confidence/coverage，避免产生无法追溯的像素。先让它们并排运行和比较，再决定默认方法。

### 6.5 P2：scene-linear RAW 与显示域分离

建立两套明确的数据域：

- `SceneLinearRGB`：配准、融合、tone model 和导出中间值；
- `DisplaySrgb`：预览、UI、JPEG/PNG 显示。

每次运行记录 camera metadata、RAW developer version、linear mode、white balance policy、lens calibration version、tone model version。混合焦段仍可以支持，但要明确哪些校正共享、哪些只能按 source 使用。

### 6.6 P2：缓存和 tiled pipeline

按 source digest、pipeline version、settings hash、proxy budget 和 geometry hash 建立 proxy cache、feature cache、station Virtual_Tile cache、render tile cache 和 Quality Gate source LRU。

所有缓存都要有 schema、provenance、原子发布、损坏降级和 eviction 指标。

### 6.7 P3：可选 learned fusion

StackMFF V3 适合做“已对齐 stack 的可选融合器”，因为它支持变量长度、任意顺序和 source probability。但当前书画输入有三个额外问题：shifted mosaic 的几何不是普通 focus stack；84 张高分辨率 RAW 与公开训练分布不同；结果必须保留严格的 source owner 和可审计证据。

因此学习模型只能放在注册完成后的 fusion stage，先输出 probability/confidence 和 source index，再经过 deterministic coverage/provenance gate。没有当前素材的训练/验证集之前，不应把 AI 输出设为唯一默认路径。

### 6.8 P3/P4：GPU 和大规模优化

等 CPU pipeline 通过 stage profiling 后，再把最适合 GPU 的部分迁移：warp/sampling、Laplacian/wavelet pyramid、focus measure、local tone field、tile copy/encode。

GPU 必须有 tiled OOM fallback 和 CPU 一致性测试。station solve、source evidence 和 Quality Gate 仍保留 CPU 可审计路径。

## 7. 推荐的对照实验

### 7.1 注册对照

固定同一组已解码 proxy，比较：

1. 当前 FAST/BRIEF；
2. OpenCV SIFT/AKAZE 或等价 Rust provider；
3. 当前特征 + ECC local refine；
4. 当前特征 + station-aware global solve。

记录候选召回、verified edge、连通分量、median/p95 reprojection error、运行时间和 peak RSS。

### 7.2 融合对照

固定同一组全分辨率已配准 source，比较：

1. 当前 owner/graph-cut；
2. Laplacian/wavelet；
3. Mertens-like quality weight；
4. DMap/edge-aware confidence；
5. OwnerHybrid；
6. 可选 StackMFF V3。

固定输出以下指标：source ownership coverage、ghost edge rate、halo rate、high-gradient seam crossing、tone_step_p95、sharpness gain、PSNR/SSIM/MEF-SSIM（有真值时）和 provenance completeness。

### 7.3 真实素材对照

- 兰亭 45 张：验证 singleton scan 路径、性能和大画布输出；
- Langyuan-10/Wenyuan-10：验证多帧 station 和 Quality Gate；
- 阆苑 84 张：验证跨机位召回、焦段变化、内存和完整覆盖；
- 枫鹰平面书画：验证 shifted mosaic、底部源分类和矩形曝光块；
- 4–16 张合成场景：验证文件名改名、导入顺序打乱、缺帧、冗余帧和错误桥。

## 8. 结论

当前项目最有价值的部分是：它没有把“生成一张看起来完整的图”当成唯一成功标准，而是已经建立了来源 owner、coverage、station relation、closure、residual、tone 和 Quality Gate 这些可审计结构。

与成熟开源和论文路线相比，主要短板不是缺少某一个神奇的融合公式，而是：

1. 注册器模式没有完全分层，导致扫描、全景和 focus stack 之间存在路径误判；
2. 特征、候选和全局优化的通用性和召回率不足；
3. 融合器缺少经典多尺度、深度图和学习方法的并排基线；
4. RAW 场景线性、缓存、tiled 和资源预算没有形成统一协议；
5. 缺少公开 benchmark 与内部真实素材之间的统一 ablation 表。

建议先完成 P0/P1：路径分类、性能观测、SourcePreflight、可替换注册器和 coarse-to-fine match graph。然后增加经典融合基线和 scene-linear RAW。GPU 与学习融合应建立在这两条基础能力稳定之后。

## 9. 外部参考

- [OpenCV Images Stitching](https://docs.opencv.org/5.0/main_modules/stitching.html)
- [OpenCV detailed stitching example](https://docs.opencv.org/4.13.0/d8/d19/tutorial_stitcher.html)
- [Hugin](https://hugin.sourceforge.io/docs/manual/Hugin.html)
- [Enblend](https://hugin.sourceforge.io/docs/manual/Enblend.html)
- [Enfuse](https://hugin.sourceforge.io/docs/manual/Enfuse.html)
- [PetteriAimonen/focus-stack](https://github.com/PetteriAimonen/focus-stack)
- [PlaneFuse](https://github.com/AdrienLF/planefuse)
- [OpenFocus](https://github.com/Xinzhe99/OpenFocus)
- [StackMFF V3](https://github.com/Xinzhe99/StackMFF-V3)
- [Brown and Lowe, Automatic Panoramic Image Stitching using Invariant Features](https://mattabrown.github.io/pdf/ijcv2007.pdf)
- [Forster et al., Complex Wavelets for Extended Depth-of-Field](https://bigwww.epfl.ch/preprints/forster0404p.pdf)
- [Mertens et al., Exposure Fusion](https://jankautz.com/publications/exposure_fusion.pdf)
- [Multi-focus Image Fusion: A Survey of the State of the Art](https://www.sciencedirect.com/science/article/pii/S1566253520303109)
