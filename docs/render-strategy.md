# 渲染层级与内存合同

本文记录 RAW Editor M1 阶段的预览/导出分辨率层级、原生缓冲所有权和确定性内存基线。
这里的 9504×6336 仍是约 60MP 的确定性尺寸算术，不是从 Sony α7R V 样片测得，也不用于声明
真实解码速度、画质或进程峰值内存。项目另有一个用户授权 α7R V 有损 ARW 的真实单样片基线；
两类证据的边界不能混用。

## 四级渲染策略

| 层级                   | 触发条件                                       | 分辨率合同                                                                                 | 输出合同                                                                                                |
| ---------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `rapidPreview`         | 滑块或其他连续交互进行中                       | `full` 保留请求尺寸；`high` 不超过大图半分辨率；`performance` 进一步降至半分辨率预算的 75% | 可见区域使用二进制 JPEG patch；原生 WGPU 可直接更新 surface                                             |
| `halfResolutionEdit`   | 交互停止、尚未达到源像素级观察                 | 上限为基础预览尺寸与源尺寸一半中的较大者，但不超过源尺寸；小图不被强制缩半                 | 完整预览或带 overscan 的 settled ROI                                                                    |
| `fullResolutionRoi`    | 交互停止、视口请求已达到源像素分辨率且存在 ROI | 输入最长边等于源图，GPU/CPU 只回读可见 ROI                                                 | ROI 使用 JPEG 4:4:4、质量 100 的二进制 patch                                                            |
| `fullResolutionExport` | 文件导出                                       | 变换与调整在完整输入尺寸执行；用户要求的 resize 和 watermark 只在处理完成后应用            | 桌面 JPEG/PNG/TIFF 逐带写文件；WebP 用完整 CPU 输入直接导入 YUVA 并有界写文件；其余格式走完整帧原生编码 |

前端 `resolvePreviewRenderPlan` 负责选择前三个层级并传递 camelCase `renderTier`；Rust
`resolve_preview_render_tier` 再次验证交互状态、ROI 和源分辨率，拒绝把
`fullResolutionExport` 送入预览 worker。导出管线固定标记为 `fullResolutionExport`，不会复用预览降采样。

所有层级继续使用同一套调整参数、WGSL 和 sRGB 合同；层级只改变处理范围、分辨率、传输和缓存，
不允许使用导出无法复现的临时图像算法。

## 缓冲所有权收敛

本阶段收敛了十四类可确定的重复缓冲或高水位资源：

1. `CachedPreview` 只保存当前层级的一张 `Arc<DynamicImage>`。快速预览直接生成目标尺寸，
   不再先生成较大底图后再长期保存一张 `small_image`。
2. 无蒙版任务绑定复用的 1×1 `R8Unorm` dummy texture，不再构造两层全尺寸零数组；有蒙版任务也不再
   把所有 bitmap 拼成第二份连续 CPU 上传数组。缺失 bitmap 的已声明层由 WGPU 资源初始化保持为零。
3. 活动蒙版按实际层数创建一个复用的 tile texture array，宽高最多为
   `(2048 + 2 × 128) = 2304`。每次图像 tile dispatch 前，`queue.write_texture` 以源 stride 和 offset
   直接上传各灰度 bitmap 的对应 ROI；WGSL 用 tile 局部坐标采样，同一层不再常驻全尺寸 GPU texture。
   CPU 侧完整灰度 bitmap 仍存在，本项没有把蒙版生成或缓存描述成流式管线。
4. 蒙版缓存保存 `Arc<GrayImage>`，命中和写入缓存时只克隆 `Arc`，当前 render 与缓存不再分别持有
   相同的全尺寸像素分配。首个有效加法子蒙版直接成为最终合成缓冲，省去初始全黑整图；黑色底图前的
   减去/相交不会生成无效临时图。每个活动蒙版仍需要一张完整 CPU 灰度 bitmap，因此本项是所有权
   收敛，不是最终结果的 tile 化存储。
5. 非空 `brush`/`clone`/`heal`/`flow` 子蒙版从第一层起按 2048px tile 生成；已有合成结果后的
   `radial`/`linear`/`all` 也逐 tile 生成，并立即向最终 bitmap 执行添加、减去或相交。生成器保留
   完整画布坐标并显式传入 tile 输出原点，避免负坐标向零截断造成 seam 偏移。每个程序化临时结果
   最多 4,194,304 B；画笔生成器额外的单笔 stroke scratch 也不超过同一 tile。本项没有消除每层
   最终 CPU bitmap。
6. `color`/`luminance` 以 2048px core 加有限 halo 生成：halo 精确等于 grow 的方形形态学半径加
   Gaussian `ceil(2σ)` 核半径，扩展区只在完整画布边缘截断，过滤后仅将 core 应用反转/透明度并
   就地合成。9504×6336 在 UI 支持的 grow `[-100, 100]`、feather `[0, 100]` 内最多为 127px halo，
   扩展 tile 不超过 2302×2302；原始 tile、形态学/模糊中间图和输出三张 scratch 合计最多
   15,897,612 B。范围匹配仍读取完整 warped source，每个最终 CPU bitmap 也仍为完整尺寸。
7. `ai-subject`/`ai-foreground`/`ai-sky`/`ai-depth`/`quick-eraser` 每个子蒙版只解码一次 base64
   灰度源，几何映射按 2048px core 加精确 halo 输出。深度范围先在 tile 内就地计算，再按 `image`
   0.25.x 的有限 Gaussian 核半径执行 depth feather，最后叠加通用 grow/feather；最大 depth 核半径
   32px 与通用 127px halo 合计 159px，三张过滤 scratch 最多为 3 × 2366² = 16,793,868 B。首个
   无过滤加法 AI 蒙版仍直接成为最终完整 bitmap；其余路径不再构造完整变换结果或完整过滤临时图。
   解码后的 AI 源和最终 CPU bitmap 仍为完整尺寸，本项不是 AI 推理或源存储的流式化。
8. WebView 回读的 RGBA8 结果使用 `Arc<DynamicImage>` 同时交给 JPEG 编码和分析 worker，
   不再为 histogram/waveform 调用 `processed_pixels.clone()`。
9. 无 patch、几何、旋转、翻转、镜头模糊或裁剪时，全尺寸变换缓存直接复用已解码图像的 `Arc`，
   不再因 `Cow::into_owned()` 复制一张完整 RGB32F 图。
10. 几何 warp 对 RGB32F/RGBA32F 输入直接借用现有底层浮点切片，采样器同时支持三、四通道；只有
    其他存储格式才回退到 RGBA32F 转换。输出合同仍是完整 RGBA32F，因此本项只消除输入 staging，
    没有把几何节点描述成 tile/带状管线。
11. 桌面 JPEG/PNG/TIFF 主图导出按横向 tile 完成一个最多 2048 行的带状 RGBA8 缓冲后，依次交给可选的
    `zenresize 0.3.1` 行 ring、可选的单行水印 scratch 和编码器；resize/watermark 不再构造完整尺寸
    `DynamicImage`。JPEG 使用 `mozjpeg 0.10.13` baseline 4:4:4 逐扫描行输入，并把压缩数据直接写入
    同目录临时文件；PNG 与 TIFF 同样直接写入。JPEG/PNG 需要保留元数据时，只在编码前生成一个受
    JPEG APP1 上限约束的 TIFF/EXIF 小载荷，再分别写入 APP1 或 PNG `eXIf`；TIFF 在编码条带前把
    筛选后的 IFD0、ExifIFD、GPSIFD 和目录指针直接写入同一输出。三种格式都不会读回压缩输出；完整
    编码、元数据写入和取消检查都成功后才原子替换目标路径。
12. CPU 回读任务只保留 tile 工作 texture；原生显示所需的 `working_texture` 与 `output_texture` 在
    该 processor 中降为 1×1。大尺寸显示 processor 转入 CPU 导出时会按 512 MiB 高水位和滞回条件收缩。
13. 批量任务全部 join 后，若 processor 与 RGBA16F 输入 texture 的逻辑占用合计达到 512 MiB，立即
    释放两者并触发一次有界 GPU poll，避免 60MP 导出的高水位常驻到后续编辑。
14. 桌面 WebP 有损编码直接把现有 RGB/RGBA 输入导入 libwebp 的 YUVA picture，不再保留第二张完整
    ARGB 工作图。libwebp 输出回调直接写临时文件，随后以固定 64 KiB 缓冲扫描/重写 RIFF、替换
    sRGB v4 `ICCP` chunk；两个阶段都成功且未取消后才原子发布。现有完整 CPU 输入与 YUVA picture
    仍然存在，本项没有把 WebP 变成 tile-to-encoder。

## 确定性前后基线

下表以 9504×6336（60,217,344 像素）计算应用代码可见的逻辑像素缓冲。MiB 使用 1,048,576
字节；除单独列出的 libwebp ARGB picture 外，这些数字不包含驱动对齐、allocator 元数据、编码器
其他内部状态或操作系统 RSS。

| 缓冲                                                   |                     修改前 |                     修改后 |          减少 |
| ------------------------------------------------------ | -------------------------: | -------------------------: | ------------: |
| 无蒙版 CPU 两层零上传数组                              | 120,434,688 B（114.9 MiB） |                        0 B | 120,434,688 B |
| 无蒙版 GPU `R8` 两层 texture                           | 120,434,688 B（114.9 MiB） |                  1 B dummy | 120,434,687 B |
| 每层活动蒙版 GPU `R8` texture                          |   60,217,344 B（57.4 MiB） |     5,308,416 B（5.1 MiB） |  54,908,928 B |
| 缓存 + caller 的每层 CPU 蒙版像素                      | 120,434,688 B（114.9 MiB） | 60,217,344 B（`Arc` 共享） |  60,217,344 B |
| 后续程序化子蒙版结果 scratch                           |   60,217,344 B（57.4 MiB） |     4,194,304 B（4.0 MiB） |  56,023,040 B |
| 画笔子蒙版结果 + 最坏单笔 stroke scratch               | 120,434,688 B（114.9 MiB） |     8,388,608 B（8.0 MiB） | 112,046,080 B |
| 颜色/明度过滤最坏三张 scratch（UI 最大 halo）          | 180,652,032 B（172.3 MiB） |   15,897,612 B（15.2 MiB） | 164,754,420 B |
| AI 深度过滤最坏三张 scratch（UI 最大 halo）            | 180,652,032 B（172.3 MiB） |   16,793,868 B（16.0 MiB） | 163,858,164 B |
| WebView 分析用 RGBA8 像素复制                          | 240,869,376 B（229.7 MiB） |          0 B（`Arc` 共享） | 240,869,376 B |
| 未变换 RGB32F 全尺寸缓存复制                           | 722,608,128 B（689.1 MiB） |          0 B（`Arc` 共享） | 722,608,128 B |
| 几何 warp 的 RGBA32F 输入 staging                      | 963,477,504 B（918.8 MiB） |          0 B（借用浮点源） | 963,477,504 B |
| 旧 Performance 1.5× 降采样的额外 RGB32F `small_image`  | 321,159,168 B（306.3 MiB） |        0 B（单一目标底图） | 321,159,168 B |
| JPEG/PNG/TIFF 最终 RGBA8 CPU 编码缓冲                  | 240,869,376 B（229.7 MiB） |   77,856,768 B（74.2 MiB） | 163,012,608 B |
| libwebp 额外 ARGB 工作图                               | 240,869,376 B（229.7 MiB） |                        0 B | 240,869,376 B |
| CPU 导出 processor 的两张 9728×6400 RGBA8 显示 surface | 498,073,600 B（475.0 MiB） |            8 B（两张 1×1） | 498,073,592 B |

这些减少项不能简单相加当作真实进程峰值：它们的生命周期并不完全重叠。回归测试只锁定每一项
不再由应用代码重新分配。GPU tile 改造本身把单层 render 的一张 CPU 灰度 bitmap + GPU texture 从
120,434,688 B 降到 65,525,760 B（减少 45.6%），而不是只剩 5,308,416 B。缓存路径在本次改造前还会
让 cache 和 caller 各持有一张 CPU bitmap；在已经使用 GPU tile 的前提下，其逻辑像素所有权从
125,743,104 B 降到 65,525,760 B，再减少 60,217,344 B（47.9%）。GPU tile texture 在整次 render
中复用，逐 tile 上传不会并发累加其容量。代价是相邻 tile 的 128px overlap 会重复传输边界像素；
本阶段只声明峰值容量和 CPU 所有权下降，不把总上传字节或处理耗时描述为同步下降。程序化像素和
画笔距离计算使用无 overlap 的 2048px tile，完整画布坐标由独立原点保留；颜色/明度则按实际
grow/feather 使用精确 halo。后续程序化子蒙版加上最终图的逻辑峰值从 120,434,688 B 降到
64,411,648 B（减少 46.5%）；
后续画笔在“子蒙版结果与单笔包围盒都覆盖全图”的最坏情况下，加上最终图从 180,652,032 B 降到
68,605,952 B（减少 62.0%）。颜色/明度在 UI 最大 halo 下，过滤 scratch 从三张完整图的
180,652,032 B 降到三张 2302×2302 tile 的 15,897,612 B（减少 91.2%）；若已有最终图，总逻辑峰值
从 240,869,376 B 降到 76,114,956 B（减少 68.4%）。AI 深度在 UI 最大 halo 下，过滤 scratch 从
180,652,032 B 降到三张 2366×2366 tile 的 16,793,868 B（减少 90.7%）。这些数字不包含仍为完整
尺寸的 warped source、AI 解码源和最终 CPU bitmap；各项生命周期不同，不能与表内其他减少量直接
相加为进程峰值。

## 自动验证

- `npm run preview-resolution:check`：验证 Retina 100%、快速预览、半分辨率编辑和全分辨率 ROI 选择。
- `npm run render-strategy:check`：锁定四级 IPC 合同、单底图缓存、dummy mask、活动蒙版 tile texture、
  CPU 蒙版 `Arc` 共享、首个加法子蒙版接管输出、程序化/画笔 2048px 分块、颜色/明度与 AI 蒙版有限
  halo、AI 源单次解码及完整坐标原点、源 offset 上传与 WGSL 局部坐标采样、分析 `Arc` 共享、原图
  `Arc` 复用、几何浮点输入借用、流式编码器边界、编码期 EXIF、禁止临时输出整文件读回、临时文件
  发布和 GPU 高水位回收。
- `npm run gpu-mask:check`：在本机 GPU 上用 2305×8 与 8×2050 确定性输入分别跨过水平、垂直
  2048px tile seam；黑色蒙版区域必须与无蒙版输出逐字节一致，白色区域必须应用曝光，seam 两侧
  分别按原始全图蒙版像素取值。该项默认忽略，避免无图形适配器的 CI 环境失败。
- Rust 单元测试：验证 camelCase 序列化、预览/导出边界、60MP 空蒙版/活动蒙版 tile 计划、蒙版局部
  坐标到全图 source offset 的一致性、首个加法子蒙版像素分配复用、黑色底图的减去/相交语义、
  radial/linear/brush/clone/heal/flow/all 在添加、减去、相交、反转和透明度下跨 seam 与完整帧逐像素
  一致，color/luminance 另覆盖旋转、翻转、粗方向、正负 grow、feather 和横纵 overlap seam；五种
  AI 蒙版再覆盖相同几何、深度双重 feather、三种组合模式、反转、透明度和横纵 overlap seam；同时
  验证空/隐藏/跨像素蒙版路由、60MP range/AI tile/halo scratch 算术、cache/caller 的 `Arc::ptr_eq`、
  全部内置 WGSL 的 Naga 解析与验证、分析结果 `Arc::ptr_eq` 共享、
  60MP 带状缓冲算术、几何 RGB32F 借用与显式 RGBA32F staging 的逐像素一致、60MP 几何缓冲计划、
  processor 收缩、JPEG/PNG/TIFF 尺寸和 sRGB v4 ICC 往返、行缩放与 batch
  参考一致、水印与旧完整帧混合逐像素一致、JPEG/PNG/TIFF EXIF 与 GPS 删除往返、TIFF → TIFF
  元数据复制，以及 JPEG
  Q50/Q75/Q92 相对旧编码器的平均 RGB 误差、文件体积和质量单调性；WebP 另验证 RGB/RGBA 在
  Q50/Q75/Q90/Q100 的新旧输出逐字节一致、ICC 往返、已有 ICC 替换、截断 RIFF、取消、原子目标
  保护和权限保留。
- `npm run synthetic-export:bench`：默认编码 9504×6336 PNG；`RAW_EDITOR_BENCH_FORMAT` 可选
  `jpeg`/`png`/`tiff`，`RAW_EDITOR_BENCH_RESIZE_LONG_EDGE=4096` 启用逐行缩放，
  `RAW_EDITOR_BENCH_WATERMARK=1` 加入确定性水印，`RAW_EDITOR_BENCH_METADATA=1` 为 JPEG/PNG/TIFF
  加入确定性 EXIF；宽高仍可通过 `RAW_EDITOR_BENCH_WIDTH` 和 `RAW_EDITOR_BENCH_HEIGHT` 覆盖。
- `npm run synthetic-webp:bench`：默认运行新的 `file` 路径；设置
  `RAW_EDITOR_BENCH_WEBP_MODE=memory` 可在同一测试二进制中运行旧的完整内存路径，宽高复用上述
  环境变量。
- `npm run synthetic-geometry:bench`：默认运行直接借用浮点输入的 `borrowed` 路径；设置
  `RAW_EDITOR_GEOMETRY_BENCH_MODE=staged` 可用同一 harness 模拟旧 RGBA32F 输入 staging，宽高复用
  上述环境变量。两种模式都保留 RGB32F 源和 RGBA32F 输出，以 10 ms 间隔采样进程 RSS。
- `npm run synthetic-mask:bench`：默认让 cache/caller 通过 `Arc` 共享 9504×6336 灰度 bitmap；设置
  `RAW_EDITOR_MASK_BENCH_MODE=cloned` 可用同一 harness 模拟旧的两份像素分配，宽高复用上述环境
  变量。两种模式都在分配前记录基线并以 10 ms 间隔采样进程 RSS。
- `npm run synthetic-mask-compose:bench`：默认以 2048px `tiled` scratch 合成确定性 9504×6336
  子蒙版；设置 `RAW_EDITOR_MASK_COMPOSITION_BENCH_MODE=full` 可模拟旧的完整临时 bitmap，宽高复用
  上述环境变量。两种模式都在最终图分配完成后记录基线，并以 2 ms 间隔采样附加 scratch 的 RSS。
- `npm run synthetic-range-mask:bench`：默认以 2048px core 加精确 halo 运行 grow=2、feather=1 的
  `tiled` 路径；设置 `RAW_EDITOR_RANGE_MASK_BENCH_MODE=full` 可运行相同输入的旧完整帧过滤。两种
  模式都保留并预先触碰最终 bitmap，以 2 ms 间隔采样形态学和 Gaussian scratch 的 RSS。
- `npm run synthetic-ai-mask:bench`：默认以 2048px core 加精确 halo 运行 depth feather=1、grow=2、
  通用 feather=1 的 `tiled` 路径；设置 `RAW_EDITOR_AI_MASK_BENCH_MODE=full` 可运行相同输入的完整帧
  几何映射与过滤。两种模式都预先触碰并共同保留完整解码源和最终 bitmap，以 2 ms 间隔隔离采样
  变换、深度 Gaussian 与 grow/feather scratch 的 RSS。

## 合成 60MP 蒙版所有权基准

2026-08-12 在同一台 Apple M5（10 核）、32 GB、macOS 26.5 上，以独立测试进程构造确定性
9504×6336 灰度 bitmap。`cloned` 模式模拟迁移前 cache 与 caller 各自持有像素，`shared` 模式使用
生产路径相同的 `Arc`；两者都让所有权存活到采样结束，并计算相同位置的稀疏哈希。

| 蒙版 cache/caller 路径 | 分配耗时 |     基线 RSS |      峰值 RSS |      RSS 增量 |  逻辑存活像素 | 稀疏输出哈希       |
| ---------------------- | -------: | -----------: | ------------: | ------------: | ------------: | ------------------ |
| 旧：两份深拷贝         |    58 ms | 11,304,960 B | 132,055,040 B | 120,750,080 B | 120,434,688 B | `aee7ceaf4a126aa6` |
| 新：`Arc` 共享         |    69 ms | 11,304,960 B |  71,794,688 B |  60,489,728 B |  60,217,344 B | `aee7ceaf4a126aa6` |

RSS 增量减少 60,260,352 B（49.9%），与一张逻辑灰度 bitmap 的 60,217,344 B 只差 43,008 B；逻辑
像素所有权精确下降 50%。两次运行的耗时顺序相反且只相差 11 ms，因此不据此声明速度提升。该 harness
不生成复杂子蒙版、不初始化 WGPU，也不代表真实 RAW 端到端峰值；它只验证缓存共享不会恢复第二份
像素分配。子蒙版生成期间的临时 bitmap 由下面的独立组合基准与逐像素测试约束。

## 合成 60MP 蒙版组合基准

同机以独立测试进程先分配并触碰一张 9504×6336 最终灰度 bitmap，再生成确定性第二子蒙版并执行
相同的 additive max 合成。`full` 模式模拟迁移前的一张完整临时结果；`tiled` 模式按生产代码相同的
2048px 上限逐块生成和就地组合。RSS 基线在最终图分配后读取，因此增量主要反映附加 scratch。

| 第二子蒙版路径      | 合成耗时 |     基线 RSS |      峰值 RSS |     RSS 增量 | 逻辑 scratch | 稀疏输出哈希       |
| ------------------- | -------: | -----------: | ------------: | -----------: | -----------: | ------------------ |
| 旧：完整临时 bitmap |    67 ms | 71,499,776 B | 132,022,272 B | 60,522,496 B | 60,217,344 B | `c5a01c6c3dfcb394` |
| 新：2048px tile     |    51 ms | 71,499,776 B |  76,251,136 B |  4,751,360 B |  4,194,304 B | `c5a01c6c3dfcb394` |

scratch RSS 增量减少 55,771,136 B（92.1%），逻辑 scratch 减少 56,023,040 B（93.0%），两条路径的
稀疏输出哈希一致。16 ms 耗时差来自合成分配图样和采样调度，不据此声明真实画笔或渐变提速。该基准
不解码 RAW、不初始化 WGPU，也不覆盖跨像素 grow/feather；可复用结论仅是已接入的 pixel-local
类型不会为了第二子蒙版再分配一张完整 60MP 临时结果。颜色/明度由下面的独立 overlap 基准约束。

## 合成 60MP 范围蒙版 overlap 基准

2026-08-12 在同机独立测试进程中先分配并触碰一张 9504×6336 最终灰度 bitmap，再对同一确定性输入
执行 grow=2、feather=1。该设置在 60MP 上同时触发 1px 形态学与 1px Gaussian 核半径，生产 halo
为 2px。`full` 使用旧完整帧过滤，`tiled` 使用生产路径相同的 2048px core、扩展 tile 和 core-only
合成；RSS 基线在最终图触碰后读取。

| 范围蒙版过滤路径        |   耗时 |     基线 RSS |      峰值 RSS |      RSS 增量 |  逻辑 scratch | 稀疏输出哈希       |
| ----------------------- | -----: | -----------: | ------------: | ------------: | ------------: | ------------------ |
| 旧：完整帧 grow/feather | 616 ms | 71,483,392 B | 258,015,232 B | 186,531,840 B | 180,652,032 B | `22b2b2054060aa43` |
| 新：2048px core + halo  | 550 ms | 71,483,392 B |  89,538,560 B |  18,055,168 B |  12,632,112 B | `22b2b2054060aa43` |

scratch RSS 增量减少 168,476,672 B（90.3%），逻辑 scratch 减少 168,019,920 B（93.0%），稀疏输出
哈希一致。66 ms 耗时差不用于声明真实颜色/明度选择提速：该 harness 直接构造灰度输入，不读取 warped
source、不执行颜色距离/明度距离匹配，也不解码 RAW 或初始化 WGPU。UI 最大 grow/feather 对应的
127px halo 已由尺寸算术和逐像素 seam 回归约束，未用最慢设置重复此 RSS 采样。

## 合成 60MP AI 蒙版 overlap 基准

2026-08-12 在同机独立测试进程中先分配并触碰一张 9504×6336 灰度解码源和一张最终 bitmap，再对
同一确定性深度输入执行几何映射、深度范围、depth feather=1、grow=2 和通用 feather=1。该设置同时
触发 1px depth Gaussian、1px 形态学与 1px 通用 Gaussian，生产 halo 为 3px。`full` 对完整帧执行
相同生产过滤，`tiled` 使用 2048px core、扩展 tile 和 core-only 合成。两种模式共同保留的源与输出
逻辑像素均为 120,434,688 B，RSS 基线在它们触碰后读取，因此增量只用于比较附加 scratch；这不是
迁移前完整调用生命周期或真实 AI 推理的端到端峰值。

| AI 深度变换/过滤路径       |    耗时 |      基线 RSS |      峰值 RSS |      RSS 增量 |  逻辑 scratch | 稀疏输出哈希       |
| -------------------------- | ------: | ------------: | ------------: | ------------: | ------------: | ------------------ |
| 完整帧对照                 | 1000 ms | 131,710,976 B | 737,411,072 B | 605,700,096 B | 180,652,032 B | `849aa825f4208721` |
| 新：2048px core + 3px halo | 1168 ms | 131,710,976 B | 201,818,112 B |  70,107,136 B |  12,656,748 B | `849aa825f4208721` |

scratch RSS 增量减少 535,592,960 B（88.4%），逻辑 scratch 减少 167,995,284 B（93.0%），稀疏输出
哈希一致。168 ms 耗时差不用于声明真实 AI 蒙版速度变化：该 harness 直接构造灰度源，不执行 base64
或 PNG 解码、ONNX 推理、RAW 解码或 WGPU。UI 最大参数的 159px halo 与 16,793,868 B tile scratch
由独立尺寸算术及五种 AI 蒙版的逐像素 seam 回归约束，未用最慢设置重复 RSS 采样。

## 合成 60MP 编码基准

2026-08-10 在同一台 Apple M5（10 核）、32 GB、macOS 26.5 上分别运行迁移前后的代码。测试构造
并重复使用与生产 GPU 拼带相同上界的 2048 行 RGBA8 缓冲，再逐行经过生产用 resize/watermark 和
JPEG/PNG/TIFF 编码器写入临时文件；RSS 由独立线程每 10 ms 采样。它不初始化 WGPU、不解码 RAW，
也不判断画质，因此不能替代真实相机验收。

| JPEG Q92 路径              | 输入 → 输出尺寸       |    耗时 |     输出大小 |      峰值 RSS | 相对迁移前峰值 |
| -------------------------- | --------------------- | ------: | -----------: | ------------: | -------------: |
| 迁移前：`zenjpeg` 收尾组装 | 9504×6336 → 9504×6336 | 1137 ms | 81,882,505 B | 330,317,824 B |              — |
| 迁移后：直接写文件         | 9504×6336 → 9504×6336 |  384 ms | 82,038,165 B |  90,308,608 B | -240,009,216 B |
| 迁移后：直接写文件 + EXIF  | 9504×6336 → 9504×6336 |  388 ms | 82,038,275 B |  90,619,904 B | -239,697,920 B |

无元数据的新路径峰值下降 72.7%，用时下降 66.2%；输出增加 155,660 B（0.19%）。同版本的全尺寸
PNG 无变换结果为 558 ms / 2,934,463 B / 90,685,440 B 峰值 RSS。EXIF 只增加 110 B 输出；其
311,296 B RSS 差异低于采样和 allocator 噪声量级，且源码合同明确禁止对临时输出调用整文件读取。
合成图样高度可压缩，输出大小与耗时不代表真实照片；可复用的工程结论是生产者带状缓冲固定为
77,856,768 B，并且 JPEG 压缩收尾和 JPEG/PNG/TIFF 元数据都不再创建与完整输出大小成正比的缓冲。
驱动对齐、GPU texture 和 RAW 解码内存仍需在端到端基准中单独记录。

## 合成 60MP 几何基准

2026-08-12 在同一台 Apple M5（10 核）、32 GB、macOS 26.5 上，以确定性 RGB32F 图样运行生产几何
warp。`staged` 模式模拟迁移前把输入转换为 RGBA32F，`borrowed` 模式直接借用 RGB32F；两者都让
原始 RGB32F 和最终 RGBA32F 输出存活到采样结束，并计算相同位置的稀疏输出哈希。

| 几何输入路径             |   耗时 |        峰值 RSS | 相对旧路径峰值 | 稀疏输出哈希       |
| ------------------------ | -----: | --------------: | -------------: | ------------------ |
| 旧：显式 RGBA32F staging | 291 ms | 2,613,788,672 B |              — | `8e4db981add93a0d` |
| 新：直接借用 RGB32F      | 180 ms | 1,649,917,952 B | -963,870,720 B | `8e4db981add93a0d` |

峰值下降 36.9%，减少量与一张 9504×6336 RGBA32F 图的逻辑大小 963,477,504 B 只差 393,216 B；耗时
下降 111 ms（38.1%）。逐像素单元测试另以 RGB32F 和显式 RGBA32F 输入锁定完整输出一致。该结果是
单次合成进程基准，不包含 RAW 解码、WGPU 或真实相机内容；可复用的工程结论仅是浮点输入不再复制，
最终 RGBA32F 几何输出仍是完整帧。

## 合成 60MP WebP 基准

同机同一测试二进制另以确定性 RGBA 图样测量桌面 WebP Q90；采样线程在分配输入图前启动，并让
完整输入保持存活到采样结束：

| WebP Q90 路径                        |      耗时 |     输出大小 |        峰值 RSS | 相对旧路径峰值 |
| ------------------------------------ | --------: | -----------: | --------------: | -------------: |
| 旧：ARGB picture + 两份内存输出      | 11,178 ms | 31,316,612 B | 1,371,684,864 B |              — |
| 新：YUVA picture + 有界文件/ICC 输出 | 11,128 ms | 31,316,612 B | 1,130,758,144 B | -240,926,720 B |

峰值下降 17.6%，50 ms（0.4%）耗时差属于运行噪声量级，最终文件逐字节一致。减少量与一张
9504×6336 RGBA8 完整帧只差 57,344 B；工程结论是直接 YUVA 导入消除了 libwebp 的第二张 ARGB
工作图，64 KiB RIFF 重写也没有恢复完整压缩输出缓冲。RSS 仍包含测试进程、完整 RGBA 输入、
libwebp YUVA picture 和编码器工作内存，不能把约 1.08 GiB 当作端到端应用预算。

## 仍然存在的峰值来源

- 全分辨率导出仍同时需要已解码/变换输入和一张 RGBA16F GPU 输入 texture；本阶段只移除了桌面
  JPEG/PNG/TIFF 的最终完整 RGBA8 输出图。几何 warp 已消除浮点源的 RGBA32F staging，但仍产生
  完整 RGBA32F 输出；RAW 解码和几何输出都尚未改造成流式节点。
- WebP 仍使用完整 CPU 输入与 libwebp YUVA picture，但已消除额外 ARGB 图和完整压缩输出缓冲；
  当前接口没有逐 tile 输入。JXL、AVIF 与 Android 导出仍使用完整 CPU 编码图，其中当前 JXL/AVIF
  编码依赖会在内部返回完整压缩 `Vec`；下一阶段优先向上游 RAW 解码和几何输出推进有界管线，并
  继续评估可替换的分块编码器。
- 桌面 JPEG 已消除完整压缩码流缓冲，但仍通过 C MozJPEG FFI 编码；当前默认 unwind 配置会把
  libjpeg 错误转换为普通导出失败，若未来改为 `panic=abort`，必须先替换这条错误边界。
- 高水位回收是导出结束和尺寸切换时的确定性策略，尚未接入操作系统/驱动的实时内存压力通知。
- 活动蒙版 GPU texture、cache/caller 所有权、程序化/画笔 scratch 与颜色/明度 grow/feather 已按
  tile、overlap 或共享所有权收敛，AI 几何变换和过滤也已接入精确 halo；但每个最终 CPU 结果仍占用
  一张全尺寸灰度 bitmap，颜色/明度匹配仍读取完整 warped source，AI 子蒙版仍保留完整灰度解码源。
  缓存预算只阻止超大条目长期驻留；后续若要继续降低蒙版峰值，需要改变最终 bitmap、范围匹配源或
  AI 解码源的存储/消费合同，而不只是继续缩小 scratch。
- 用户授权的 α7R V 有损 ARW 已建立一次本机解码、CPU 预览和全尺寸 JPEG 耗时基线；应用内首屏、
  滑块 P95、100% GPU ROI、GPU 文件导出和进程峰值，以及其余光照/ISO/压缩模式仍待新样片扩充，
  不得用上述尺寸算术或单样片结果替代。
