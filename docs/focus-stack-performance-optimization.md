# 堆栈性能方案与历史验收记录

> [返回项目进度与任务总览](project-progress.md#stack) · 专项文档。本文保留性能技术方案与逐轮验收记录，历史P0–P4、阶段0–8及“下一步”不再作为另一份当前任务表。项目状态与执行顺序只在总览维护。

本文面向 `raw-editor` 当前的 Rust 堆栈管线，目标是降低真实 RAW 堆栈的等待时间和内存峰值，同时保持现有的几何证据、来源可追溯性、`Station_Relation` 门禁、`Quality_Gate` 语义和输出稳定性。

本文按实际落地顺序组织。每一阶段都必须先获得可比较的基线，再进入下一阶段；任何“变快”都不能只用进程退出码或生成了一张图来证明。

## 1. 目标和边界

### 1.1 目标

1. 减少同一张 Source_RAW 的重复读取、重复 EXIF 解析和重复 RAW 解码。
2. 减少没有希望的 pair 进入完整描述子匹配和 RANSAC 的次数。
3. 减少 Virtual_Tile、render 和 Quality Gate 之间的重复渲染。
4. 控制并发解码、Rayon 嵌套并行和图割内存，使运行不进入 swap 抖动。
5. 让每次运行都能回答“慢在哪里、命中了什么缓存、使用了多少内存”。

### 1.2 不改变的内容

以下内容是性能优化的硬约束：

- `Station_Relation` 的真实对应点、对称误差、尺度、空间覆盖、光度和边缘门禁；
- `Quality_Gate` 的阈值和 `passed`、`diagnostic-only`、`rejected` 语义；
- RAW 生产加载路径，不能用通用 `image::open` 替代 NEF 解码；
- 单源 owner、Coverage、Ownership、Provenance 和最终来源可追溯性；
- 文件名只能作为稳定排序依据，不能单独创建几何关系；
- 现有输出格式、颜色语义和失败时的取消/拒绝行为。

### 1.3 优化原则

- 先观测，再改行为。
- 先消除重复工作，再调整并行度。
- 先保留现有完整质量路径，再把快速路径限定在分析或预览阶段。
- 每个优化都必须有独立开关、独立指标和可回退路径。
- 真实数据必须同时做冷启动和热启动测试。

## 2. 当前慢点和优先级

| 优先级 | 热点 | 当前行为 | 主要风险 |
| --- | --- | --- | --- |
| P0 | 运行路径选择 | 45 张移动扫描被识别为多帧 station 时，会走 `layered_virtual_tile` | 图割、Virtual_Tile 关系和多次 tile 渲染会把分钟级运行推到几十分钟 |
| P0 | Source preflight | 焦段、尺寸、曝光、SHA-256、RAW 解码分多次打开同一文件 | 大 RAW 重复读盘，页缓存和等待时间增加 |
| P0 | 分析/渲染重复解码 | scalable 路径不保留 `full_image`，`load_render_image` 和 Quality Gate 可能再次解码 | 同一源图重复消耗 RAW demosaic 和内存 |
| P1 | Pair 数量 | 小堆栈 FocusStack 直接 exhaustive；45 张是 990 对，84 张是 3,486 对 | 每对都进入完整 matcher/RANSAC，耗时近似平方增长 |
| P1 | Pair 内存 | matcher 为特征对创建完整距离矩阵；外层 pair 和内层 RANSAC 还可能嵌套并行 | 临时分配过多，RSS 抖动，线程越多越慢 |
| P1 | Virtual_Tile 缓存 | `VirtualTileStore` 已存在，但生产路径尚未形成稳定的跨运行命中链路 | 重复运行仍重复 fuse station |
| P2 | 图割和 tile | 大量 label、全分辨率扫描、候选和最终输出之间重复采样 | 单机位或大画布场景下 CPU 和内存占用高 |
| P3 | GPU | 当前瓶颈位置尚未通过完整计时确认 | 过早 GPU 化会把 IO、解码或候选搜索问题掩盖起来 |

## 3. 阶段 0：建立可比较的性能基线

这一阶段不改变像素和几何行为，只增加观测。

### 3.1 Stack_Report 新增字段

建议在现有 report schema 下增加或补齐 `performance`：

```json
{
  "performance": {
    "stage_ms": {
      "preflight": 0,
      "prepare": 0,
      "match": 0,
      "group_solve": 0,
      "station_solve": 0,
      "tile_fuse": 0,
      "render": 0,
      "quality_gate": 0,
      "total": 0
    },
    "pairs": {
      "candidate": 0,
      "coarse_pass": 0,
      "verified": 0,
      "rejected": 0
    },
    "cache": {
      "source_digest_hit": 0,
      "proxy_hit": 0,
      "feature_hit": 0,
      "render_hit": 0,
      "virtual_tile_hit": 0,
      "virtual_tile_miss": 0,
      "bytes_read": 0,
      "bytes_written": 0
    },
    "resources": {
      "peak_rss_bytes": 0,
      "max_concurrent_raw_decode": 0,
      "rayon_threads": 0,
      "swap_risk": false
    },
    "throughput": {
      "source_pixels_per_second": 0.0,
      "output_pixels_per_second": 0.0
    }
  }
}
```

字段命名可以按当前 schema 调整，但必须保留阶段边界和缓存命中率。不要只记录总耗时。

### 3.2 埋点位置

优先在以下函数周围使用 `Instant` 埋点：

- `stitch_images_with_options`：总耗时、预检、准备、匹配、分组和最终渲染；
- `load_prepared_stack_source`：文件读取、RAW decode、RAW enhancement、镜头校正；
- `pairs_to_match_for_images`：候选 pair 数量；
- `match_image_pair`：coarse、完整 matcher、RANSAC 和 reject reason；
- station grouping/solver：分组、拓扑、pose 和 closure；
- `focus_stack_stitcher`、`layered_virtual_tile_compositor`：tile 数量、融合、图割和重渲染次数；
- `quality_gate_runner`：Quality Gate 自身耗时和 `decode_seconds`；
- `load_render_image` 和 `quality_loader`：每源加载次数、缓存命中和解码耗时。

### 3.3 基线矩阵

每组都要记录源码 commit、输入文件清单和输入 hash：

| 场景 | 目的 | 运行方式 |
| --- | --- | --- |
| 4–8 张合成/本地小包 | 快速验证行为和计时字段 | 普通库测试和 targeted test |
| 兰亭 45 张，10% | 快速比较路径选择、候选和 tile 数量 | preview-only，串行 |
| 兰亭 45 张，1:1 | 检查真实解码、完整渲染和 RSS | 真实 fixture，串行 |
| Langyuan-10 / Wenyuan-10 | 多帧 station 和 Quality Gate 回归 | 1:1，串行 |
| 阆苑 84 张 NEF | 最终压力和完整素材合同 | 只在前面阶段稳定后执行 |

当前 45 张移动扫描的历史基线显示，误走 layered 路径时 10% 运行约 25 分钟，而正确回到 legacy streaming 路径约 193 秒；这个差异必须作为第一阶段的路径选择回归项，而不是当作普通线程调优问题。

### 3.4 阶段 0 通过条件

- report 能区分所有阶段耗时；
- 能记录 candidate、verified、Virtual_Tile 数量；
- 能记录 peak RSS、最大同时 RAW decode 数；
- 同一输入连续运行两次，报告中的 source hash 和几何结果一致；
- 输出像素、输出 hash、Quality Gate 结论与现有基线一致。

### 3.5 当前落地状态

阶段 0 的观测已落地：`Stack_Report.performance.stage_seconds` 记录 `preflight`、`prepare`、`matching`、`ordering`、`render` 和 `total`；`performance.pairs` 记录基础邻接/全量 pair、compact retrieval 新增 pair、最终 candidate pair、verified pair 以及 rescue 新增 pair。这样可以区分候选筛选减少和后续几何验证变慢，避免只看一个总 pair 数。

阶段 2 的 SourcePreflight 也已接入 FocusStack 运行入口。一次运行通过同一份映射字节完成源文件映射、SHA-256、尺寸、EXIF、焦距和 RAW 曝光元数据采集；cache key、准备、rescue、render、PSD 扫描和 Quality Gate loader 优先复用这份记录。`performance.source_preflight` 记录 `stat_count`、`file_map_count`、`hash_count`、`exif_parse_count`、`preflight_reuse_count`、`raw_decode_count` 和失败数。这里的 `raw_decode_count` 是实际完整图像解码次数，仍可能因为 bounded-memory 策略和 Quality Gate 的来源证据需求大于源数量；后续受限 LRU 和 Quality Gate source/ROI 复用必须以这个计数为验收依据，不能把映射复用误报成 RAW 解码复用。

当前已补齐 SourcePreflight 的 PNG fixture 单测、报告观测写盘单测、性能字段默认值和 JSON round-trip 检查；真实 45 张 NEF 的 RSS 与冷/热运行也已执行。由于该数据集在几何硬门处拒绝，Virtual_Tile 跨运行缓存和 Quality Gate ROI 命中仍待一个连续重叠素材验收，不能用定向单测替代。

## 4. 阶段 1：先修正运行路径选择

这一阶段优先于通用缓存，因为错误路径会放大后续所有成本。

### 4.1 目标

对移动扫描、长序列或大多数 station 为单帧的输入，直接使用已有的 `legacy_single_layer_mosaic`/streaming 路径；只有确实存在多帧 focus station 且需要逐像素 owner 决策时，才进入 layered Virtual_Tile 路径。

### 4.2 实施位置

重点检查：

- `capture_groups_form_singleton_scan`；
- focus grouping 的 singleton 判定；
- `select_and_record_run_path`；
- `focus_stack_stitcher` 的 compositor 选择；
- `stack_report.selected_path` 和 `virtual_tile` 标记。

### 4.3 判定要求

1. singleton 判定只能使用已验证的 group/station 证据，不能按文件名或固定素材名称判断。
2. 记录 `station_count`、`singleton_station_count`、`multi_frame_station_count`、`isolated_source_count`、configured path、最终 selected path、`use_virtual_tiles` 和 `fallback_reason`。
3. 多帧 station 仍保留 layered 路径。
4. selected path 改变时，必须保留完整报告并明确记录原因。

### 4.4 通过条件

- 兰亭 45 张 10% 和 1:1 默认均回到 `legacy_single_layer_mosaic`；若后续显式启用 `streaming_mosaic`，必须作为独立 comparison path 记录；
- 45/45 source 被保留，覆盖、笔画、印章和色调边界与基线一致；
- Langyuan-10/Wenyuan-10 仍进入原有 layered 路径；
- 没有通过放宽 station grouping 或降低 Quality Gate 阈值来获得路径切换；
- 10% 与 1:1 的 selected path、group 成员和关系集合稳定。

### 4.5 诊断输出开关

`stack_diagnostics.output_dir` 默认关闭。性能基线期间只保留 Stack_Report 和必要的 ROI；不要在普通运行中为每个 station 写 members、transforms、residual、ownership、confidence、coverage 的完整 JSON。诊断开启时要在 report 记录 `diagnostics_enabled=true`，避免把诊断写盘时间误认为算法耗时。

## 5. 阶段 2：统一 SourcePreflight 和运行上下文

### 5.1 目标

让每个源图在一次受控的 preflight 中完成文件身份、尺寸和必要元数据采集，后续阶段只消费结果，不重复打开文件。

### 5.2 建议数据结构

在 `run_control.rs` 或新的 `source_context.rs` 中定义运行内结构：

```rust
struct SourcePreflight {
    path: PathBuf,
    size_bytes: u64,
    modified_ns: Option<u128>,
    file_identity: Option<FileIdentity>,
    sha256: Option<[u8; 32]>,
    width: Option<u32>,
    height: Option<u32>,
    focal_length_35mm: Option<f64>,
    camera_make: Option<String>,
    camera_model: Option<String>,
    exposure_value: Option<f64>,
    is_raw: bool,
}

struct RunSourceContext {
    sources: Vec<SourcePreflight>,
    raw_exposure_reference: Option<f64>,
    pipeline_version: String,
}
```

不建议把所有源文件的 mmap 永久保存在 context 中。可以在单个 preparation worker 内复用 mmap，或只在明确的 source LRU 中保留有限数量的映射。

### 5.3 建议流程

1. `stat` 文件并记录 size、mtime 和平台文件身份。
2. 读取一次 mmap，在同一份 bytes 上完成 SHA、EXIF、焦段、相机和曝光读取。
3. 对能从 RAW metadata 得到的尺寸直接使用 metadata，不再调用额外的 `image_dimensions`。
4. 用 context 计算 `selection_has_mixed_focal_lengths`、`stack_raw_exposure_reference` 和 cache key。
5. `load_prepared_stack_source` 接收 preflight 结果；需要重新读取时先检查文件身份是否变化。
6. run 结束的 digest recheck 仍保留在严格验收模式。普通交互模式可以先比较 size/mtime/file identity，再决定是否需要完整重 hash，但不能在验收模式跳过完整复核。

### 5.4 缓存失效

source metadata/digest cache 的 key 至少包含：

```text
absolute_path
file_identity
size_bytes
modified_ns
```

以下变化必须导致 miss：

- 内容 hash 变化；
- RAW 处理设置变化；
- 线性模式、highlight compression、降噪或 sharpening 变化；
- pipeline version 变化。

### 5.5 通过条件

- 单次运行中每个源的 preflight 文件读取次数为 1（取消和失败重试单独记录）；
- `selection_has_mixed_focal_lengths`、曝光参考和 cache key 不再各自读完整文件；
- strict acceptance 的 source hash 与原路径一致；
- 结果、报告和拒绝原因与基线一致；
- 45/84 NEF 的 `bytes_read` 和 `preflight_ms` 明显下降。

## 6. 阶段 3：分离分析代理图和最终渲染

### 6.1 目标

配准阶段只生成足够用于特征和几何的代理图；最终输出仍使用完整质量 RAW。

### 6.2 分析阶段

对 `load_prepared_stack_source` 增加明确的分析模式：

- bounded stack：默认长边 1,536–2,400px；
- fast demosaic 或嵌入式 preview 只用于 proxy；
- 直接从 RGB32F 生成 luma/downsample，避免 `to_rgb8 -> grayscale -> resize` 的重复整帧转换；
- 特征、foreground mask、edge candidates 和 top features 只消费同一个 proxy；
- proxy 生成后立即释放不需要的中间色彩图。

### 6.3 最终渲染阶段

- 只有进入最终画布的 source 才完整解码；
- 如果 `DynamicImage` 已经是 RGB32F，使用 move/into 转换，避免再次 clone；
- 预先计算的 shared RAW display gain 和 lens calibration 结果必须复用；
- 最终渲染不得使用 fast demosaic 代替 full quality RAW。

### 6.4 Proxy/feature cache

proxy 和 feature cache 的 key：

```text
source_sha256
pipeline_version
raw_processing_settings_hash
proxy_width
proxy_height
proxy_color_encoding
brief_pairs_version
feature_detector_version
```

缓存内容建议包括 proxy luma、feature keypoints、BRIEF descriptor、foreground/edge 辅助数据。缓存不得包含无法验证来源的生成像素。

### 6.5 通过条件

- 配准几何、group 成员、candidate/relation 集合与基线一致；
- 最终 full render 的 RGB/alpha、来源 owner 和 Quality Gate 结果不变；
- 取消请求不会留下半发布的 proxy 或 render cache；
- 热启动可以跳过 proxy 和 feature extraction，并在 report 中记录命中率。

## 7. 阶段 4：两阶段候选匹配

### 7.1 目标

减少没有希望的 pair 进入完整 matcher 和 RANSAC，同时保留跨文件名、跨焦段和跨组件的真实桥接机会。

### 7.2 候选阶段

第一阶段只使用低成本数据：

1. 输入顺序邻接 pair；
2. natural filename 邻接 pair，仅作为稳定候选来源；
3. compact feature 子集的 mutual descriptor count；
4. 可选的粗 NCC、尺度桶或 LSH/倒排索引；
5. 每个 source 保留 top-K 视觉候选，K 初始取 6–12，并保留必要的邻接补边。

### 7.3 完整验证阶段

只有 coarse pass 的 pair 才调用当前完整流程：

- ratio matching；
- mixed-focal scale candidate；
- RANSAC；
- spatial support、scale、overlap、photometric 和 edge gate；
- station relation 和 closure gate。

对图不连通的结果，按连通分量边界追加 bounded rescue；不要退化成无上限的全对搜索。

### 7.4 特征匹配内存优化

`match_features_with_ratio` 应逐步从完整 `features1 × features2` 距离矩阵改为：

- 分块计算；
- 每行只保留 top-2/top-4；
- 需要反向 best 时保留紧凑索引而不是完整矩阵；
- 对 mixed-focal 的多种 ratio/scale gate 复用一次距离排序结果。

### 7.5 通过条件

- `pair_candidates` 和完整 `verified_pairs` 明显下降；
- 正例、负例、跨焦段桥、无重叠 pair 的拒绝结果不变；
- 84 张场景不出现因为粗筛漏掉真实跨组件桥而导致的断连；
- 仍满足现有 Station_Relation 硬门：真实对应点、误差、尺度、凸包和光度/边缘证据；
- 文件名改名、输入顺序打乱后，group 成员和输出覆盖稳定。

## 8. 阶段 5：接入 Virtual_Tile、render 和 Quality Gate 缓存

### 8.1 Virtual_Tile 持久化缓存

将 `VirtualTileStore` 从测试可见路径提升到生产使用路径，按 station 维度加载和存储。

缓存 key 必须包含：

```text
pipeline_version
sorted_source_paths
sorted_source_sha256
proxy/render budget
raw settings hash
station geometry hash
tone/residual model version
```

缓存条目必须保存并校验：

- pixels；
- coverage；
- ownership；
- confidence；
- provenance；
- geometry/canvas dimensions；
- source SHA-256。

命中后可以跳过 station fusion，但仍要校验 geometry、来源列表和缓存 schema。缓存失效、损坏或不完整时必须安全回到 miss 路径。

### 8.2 单次运行 LRU

在磁盘缓存之外，增加受限的 per-run LRU：

- source render image；
- render tile；
- Quality Gate 需要的 owner source；
- proxy/feature 结果。

LRU 必须有像素或字节上限，达到上限后按 least-recently-used 释放；report 记录 resident peak 和 eviction 次数。

当前落地先采用 Quality Gate ROI 级 LRU：上限为 64 MiB，条目保存已经完成 tone 变换的 `Rgb32F` 参考 ROI，key 包含源路径、几何矩阵、native registration/warp 标记和 ROI 世界坐标。这样不会把 45 张完整 RAW 常驻内存，也不会因为跨运行复用旧 RAW 而绕过 SourcePreflight。`Stack_Report.performance.quality_roi_cache` 记录 hits、misses、evictions、resident/peak bytes 和 resident entries；`quality_gate.timing` 同步记录本次 gate 看到的累计计数。

完整 Source_RAW 图像仍按 bounded-memory 策略在 render 与首次 Quality Gate 之间释放；本阶段不把 `DynamicImage` 直接放进一个无界缓存。若后续要扩展到 source image LRU，必须先加入字节预算、RSS 保护和真实 45 张 NEF 的命中收益证据，再改变该边界。

### 8.3 Quality Gate 复用

Quality Gate 当前可能通过 `quality_loader` 再次完整解码 RAW。优先复用 render 阶段已有 source/LRU；若 Quality Gate 只需要稀疏 ROI，则从已生成的 owner tile 或受控 proxy 中取样，但必须保持现有 Quality Gate 的证据语义。

当前实现把同一个 ROI LRU 贯穿首次 Quality Gate 与 boundary repair 后的重跑：首次 gate 生成 post-tone ROI，重跑时命中即可跳过该 owner 的完整 RAW 解码；当一个 owner 的 ROI 仍有 miss 时，只对该 owner 解码一次，并继续记录 `owner_source_decode_count`。因此验收时要同时看 `performance.source_preflight.raw_decode_count`、`quality_gate.timing.owner_source_decode_count` 和 ROI cache 命中/淘汰，不能只看文件映射或 preflight reuse。

禁止为了省一次 decode 删除 Quality Gate 或减少其必要 ROI。

### 8.4 通过条件

- 重复运行出现可解释的 cache hit；
- 命中时输出、coverage、ownership、provenance 和 Quality Gate 与 miss 路径一致；
- 修改一张输入 RAW 后只影响相关 cache key；
- 删除/损坏 cache 后能自动 miss 并成功重建；
- cache 写入采用临时目录、完整校验、原子 rename，不影响最终输出发布。

### 8.5 本轮 P1 验收记录（2026-10-31）

已对真实 `sleeping` 数据集执行 45 张 NEF、`scale=1`、`--write-output` 的 cold/hot 两次运行。数据契约为 45 个源文件、约 2.56 GB，父目录为 `/Users/dp/Downloads/书画拼图`，运行报告分别落在：

- cold：`/private/tmp/raw-editor-regression/sleeping-s1-p1-cold-stack-reports/stack-report-61f5ad54-1804-4731-a6a2-8e0be49520b9.json`；
- hot：`/private/tmp/raw-editor-regression/sleeping-s1-p1-hot-stack-reports/stack-report-250736f5-f7f7-4a51-b439-91df866eee31.json`。

两次结果都被现有几何硬门拒绝：15 个 station 最终形成 5 个不连通分量，`quality_gate.verdict=not_run`，因此本轮真实数据没有产生 ROI cache hit；这不是 Quality Gate 通过/失败结论。关键计数如下：

| 运行 | total | preflight | prepare | matching | candidate/verified | raw decode | cache |
|---|---:|---:|---:|---:|---:|---:|---|
| cold | 1491.48 s | 0.76 s | 58.19 s | 26.15 s | 990 / 661 | 90 | 0 / 0 / 0 |
| hot | 1501.71 s | 0.37 s | 77.26 s | 26.78 s | 990 / 661 | 90 | 0 / 0 / 0 |

这里的 hot 只是同一进程外的第二次运行，不能命中 per-run ROI LRU；它只用于观察操作系统页缓存和重复运行的阶段计时。`raw_decode_count=90` 高于 45 个 source，是当前失败路径中准备阶段与后续 Virtual_Tile/render 取源各解码一次的实测证据；后续应优先围绕几何分组/候选搜索修复后再测完整 Quality Gate ROI 复用。

针对这条计数链路，已将 bounded-memory rescue 改为直接从初始准备阶段保存的 `ImageInfo.alignment_image` 重建严格特征视图（仍使用 2,400px 和 1,600 features），不再为进入 disconnected-component rescue 的源再次完整解码 RAW。该改动已通过 `source_preparation` 定向测试（12 passed）和格式/差异检查。随后重新执行了真实 45 张 NEF 的 `p1-rescue` 运行，但 `raw_decode_count` 仍为 90、`preflight_reuse_count` 为 90，说明这次数据路径中的第二批 45 次解码来自后续 Virtual_Tile/render 取源，而不是 rescue 本身；这也把下一步瓶颈从“rescue 重复解码”收敛为“render 到 Quality Gate 之间的受限 source 复用”。当前仍只保留 64 MiB 的 post-tone ROI LRU；在没有命中收益和 RSS 证据前，不扩展成无界 source image cache。

`p1-rescue` report：`/private/tmp/raw-editor-regression/sleeping-s1-p1-rescue-stack-reports/stack-report-0c0a115d-512b-4694-b26a-107fae10a6a1.json`。本次总耗时 `1496.95s`，preflight `0.82s`、prepare `61.64s`、matching `26.41s`，candidate/verified `990/661`；仍是 15 个 station、5 个不连通分量，`quality_gate.verdict=not_run`，ROI cache `0/0/0`。native harness 因既有 `geometry_disconnected` 硬门而拒绝，测试进程的失败退出码是预期的几何拒绝证据，不能当作编译或 Quality Gate 失败。

这次 `p1-rescue` 是在 pair plan 字段加入前启动的，因此 JSON 中仍只有 `candidate_pairs/verified_pairs`；后续新报告会同时写入 `base_pairs/retrieval_pairs/final_pairs/rescue_pairs`，不改变匹配决策。

本轮几何复核进一步确认：45 张 `sleeping` NEF 并非一个可安全串成单一几何图的连续场景。15 个 station 形成 5 个不连通分量（source counts 为 `2/19/10/5/9`）；105 个 station candidate 已全部检查，没有候选预算截断。跨分量仅有 4 条可拟合候选，尺度分别为 `0.9063`、`0.9097`、`0.9233`、`0.9350`，全部被既有 `0.95–1.05` Station_Relation 硬门拒绝，其余候选主要是 `polished_fit_failed` 或 `insufficient_prefit_hull_support`。因此 `geometry_disconnected` 是有效拒绝证据，不能通过降低尺度门槛、按文件名补 bridge 或把 Quality Gate 标记为通过来换取 ROI 命中；后续 ROI cold/hot 验收应使用一个真实连续重叠的数据子集或按分量分别运行。

### 8.6 连续真实子集复核与 bounded-memory 回归

历史 `langyuan-10` 报告中的 `DSC_3680–3689.NEF` 确实是连续重叠素材，并曾在旧 pipeline 中得到 3 个 station（4/3/3）和 12 个 Quality Gate ROI。但用当前严格 station link 直接重跑时，NEF 被错误地走到了 full-resolution preparation，每张只产生约 18–46 个特征；当前门禁要求至少 30 个一致对应点，因此这些源会被记录为 `grouping_no_overlap_evidence`，不能进入 Quality Gate。`DSC_3687–3689.NEF` 和 `DSC_3700–3703.NEF` 的单独运行也复现了这一现象；后者虽有 34–39 个 pair inliers，但被判定为 singleton source scan，按语义不应运行 Quality Gate。

根因是 SourcePreflight 的尺寸探测对 RAW 容器可返回尺寸，`requires_bounded_memory()` 因而跳过了旧路径对 RAW 的保守 bounded-memory 判定。修复后，任何 RAW 选择都会进入 bounded preparation，恢复约 2,400px 对齐平面和 1,600 特征上限；PNG/JPEG 仍按像素总量走原有判断。该修复补有 `.nef` fixture 单测，验收时必须先确认日志出现 `bounded-memory mode`，再比较 station、ROI 和 decode 计数。

下一轮真实验收优先采用 `sleeping` 中同一几何连通分量的子集，而不是把 45 张不连通素材混在一起。候选分量为 `DSC_6483–6487.NEF` 与 `DSC_6479–6482.NEF`（9 张、3 个 station、2 条已接受跨 station relation），其次是 19 张的大分量。cold/hot 两次运行均需使用同一子集、`scale=1`、`--write-output`，并同时记录：

- `grouping.station_count`、`station_relations.accepted` 和最终 `quality_gate.verdict`；
- `performance.source_preflight.raw_decode_count`、`preflight_reuse_count`；
- `performance.source_render_cache` 的 `max_bytes`、总 hits/misses，以及分阶段的 `render_hits/render_misses`、`quality_gate_hits/quality_gate_misses`、evictions、resident/peak bytes 和 oversized bypass；分阶段计数用于证明命中是否真正发生在 render→Quality Gate，而不是只发生在多个 virtual tile 的 render 内部。
- `quality_gate.timing.owner_source_decode_count`、`roi_reference_count` 与 `performance.quality_roi_cache`。

hot 仍指同一进程之外的第二次运行，不能期待命中 per-run LRU；它用于观察文件系统页缓存和重复运行的阶段计时。render→Quality Gate source 复用只允许发生在同一运行内，key 必须同时绑定路径、SourcePreflight SHA、输出尺寸和像素处理 fingerprint，缓存采用 512 MiB 字节上限，单个超限源直接 bypass，取消或 RSS 超阈值时清空。验收或受控 profile 可以通过 `RAW_EDITOR_STACK_SOURCE_RENDER_CACHE_MAX_BYTES` 临时提高上限（代码内仍限制到 4 GiB）；产品默认仍为 512 MiB。这样可以在明确的 RSS 预算下验证全尺寸 source→Quality Gate 命中，而不会把默认路径变成无界缓存。

本轮还补上了 Quality Gate ROI 选择的最后一档 bounded fallback：小平面路径在 512/256/128px 都找不到满足 owner/coverage margin 的窗口时，会尝试 64px；大平面 sparse selector 也保持同样的 64px 最后 fallback。16px owner/coverage margin、13px acutance 支持和全部 Quality Gate 阈值没有改变，只有可测证据窗口的尺寸允许下降。对应的属性测试和 interleaved-owner 定向测试已覆盖该分支。

真实子集的当前验收证据如下：

- 9 张连续组件、`scale=0.5`、preview-only：`/private/tmp/raw-editor-regression/sleeping-9-roi-cold-s05-stack-reports/stack-report-3156bb7e-6f17-4af1-8bc4-36dbe19dae42.json`。`station_count=3`、连通分量为 1、两条 Virtual_Tile relation 被接受；`source_render_cache` 为 `hits=30`、`misses=87`、`evictions=84`、`resident_entries=3`，说明受限 cache 在同一次 render 中确实复用。但 `quality_gate.roi_count=0`、`verdict=insufficient_evidence`，且 `closure.status=no_constraints`，因此不能把这次运行称作 QG pass。
- 4 张连续组件、两个 station、`scale=0.5`、preview-only：`/private/tmp/raw-editor-regression/sleeping-4-two-station-s05-roi64-v2-cold-stack-reports/stack-report-1d6c57ef-c419-414c-ab86-c5241173755c.json`。几何连通分量为 1，cache 为 `hits=20`、`misses=16`、`evictions=13`，但最终仍没有可测 ROI，`quality_gate.verdict=insufficient_evidence`。该证据表明真实 NEF 的 owner runs 仍短于 64px，下一步应改进“从 station-level owner 证据派生 ROI”的算法，而不是继续扩大完整 RAW cache。

随后将同一 4 张源切回全尺寸 `scale=1`，并在 preview-only 下分别做了默认预算和受控 2.4 GB 预算的对照。默认 512 MiB 报告 `/private/tmp/raw-editor-regression/sleeping-4-two-station-s1-roi64-cold-stack-reports/stack-report-204943f9-3196-4214-a409-ff710188e49e.json` 已有 `roi_count=71`、`quality_gate=pass`，但 4 张全尺寸 source 均超过预算，`render_hits=0`、`quality_gate_hits=0`、`oversized_bypasses=38`，`raw_decode_count=42`。在明确设置 `RAW_EDITOR_STACK_SOURCE_RENDER_CACHE_MAX_BYTES=2400000000` 后，cold/hot 报告分别为：

- cold：`/private/tmp/raw-editor-regression/sleeping-4-two-station-s1-cache24-cold-stack-reports/stack-report-31df3160-9f1b-4cf8-8f96-cb33126375a6.json`；
- hot：`/private/tmp/raw-editor-regression/sleeping-4-two-station-s1-cache24-hot-stack-reports/stack-report-6fd97d65-aa10-4e83-967e-e12bb72158eb.json`。

两次都得到 `quality_gate.verdict=pass`、`roi_count=71`、`owner_source_decode_count=2`、`quality_gate_hits=2`、`quality_gate_misses=0`、`render_hits=32`、`render_misses=4`、`raw_decode_count=8`，且 preview 与 metrics SHA-256 一致。cold/hot 的 total 分别为 `243.83s/230.53s`，hot 的 prepare/matching 由 `18.08s/5.67s` 降到 `12.15s/2.46s`；这部分是操作系统页缓存收益，per-run source LRU 每次都会从空 cache 开始。2.4 GB 是验收 profile 的明确上限，不是默认产品预算；默认仍以 512 MiB 的 oversized bypass 保护内存。

这两份真实报告是在 `max_bytes` 字段加入 Stack_Report 前生成的，因此报告中用 `resident_bytes=2,211,124,992` 和 `peak_bytes=2,211,124,992` 证明实际受控上限；当前代码已经把 `max_bytes`、分阶段 hits/misses 和 resident 统计全部序列化，后续验收应优先检查新字段。

因此当前阶段的结论是：受限 render cache 已落地，默认预算保护路径和受控大预算路径都有真实 Stack_Report 证据；在 2.4 GB 明确上限下，render→Quality Gate 已完成 cold/hot 验收并得到 2 次 Quality Gate source 命中。`quality_gate_hits/quality_gate_misses` 是后续验收的必查字段，不能只看总 hits。64px ROI fallback 解决了全尺寸大平面的可测性，但 `scale=0.5` 的小平面仍可能因 owner runs 过短而无 ROI，这个限制要保留在后续 station-level ROI 算法任务中。

### 8.7 station-level owner evidence 修复后的 9 张 NEF 冷/热验收（2026-11-02）

针对上述 `scale=0.5` 的 ROI=0 问题，本轮完成了四项实现修复：station evidence selector 使用 256px primary window；水平 owner runs 建索引后再检查候选窗口；station-level ROI 固定保留至少 8 个可测窗口并优先于 raw-owner lattice；22M 像素以上的验收画布走 streaming Quality Gate boundary 路径，并在跨 station boundary 的 64px 空间单元内做有界去重。这些改动不改变 Quality Gate 阈值，只限制观测候选数量和选择路径。

修复后的真实输入为 `DSC_6479.NEF`–`DSC_6487.NEF` 共 9 张，`scale=0.5`，受控 source-render cache 上限为 2.4 GB。最终 Stack_Report：

- cold：`/private/tmp/raw-editor-regression/sleeping-9-owner-derived-s05-cache24-cold-v10-reports/stack-report-e143d9d6-90f3-4ec1-8174-83d261d6707d.json`；
- hot：`/private/tmp/raw-editor-regression/sleeping-9-owner-derived-s05-cache24-hot-v10-reports/stack-report-c6c31a35-2964-4f77-9e8b-20c83e7789d9.json`。

两次报告都确认 station-level evidence 已生效：`grouping.station_count=3`、接受 2 条 station relation、`quality_gate.roi_count=8`、`station_owner_evidence_roi_count=8`、`raw_owner_evidence_roi_count=0`。QG 的 `verdict=fail` 是 criteria/unmeasurable 结果，已经不是 ROI 选择失败；`unmeasurable` 记录数为 141。两次 native closure 都是 `no_constraints`，所以现有 harness 在 `focus_stack_fixture.rs:570` 以 geometry closure panic 退出；该退出必须和 Quality Gate verdict 分开记录。

| 运行 | total | render | QG ROI | station evidence | QG timing | cache hits/misses | QG source hits | raw decode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | 645.44 s | 619.43 s | 8 | 8 | 2.506 s | 120 / 9 | 12 | 18 |
| hot | 645.45 s | 619.86 s | 8 | 8 | 2.843 s | 120 / 9 | 12 | 18 |

QG timing 的 cold/hot 细项分别为：reference resampling `1.899/2.118s`、local scale `0.00046/0.00049s`、MTF `0.0767/0.0848s`、gradient `0.3967/0.4807s`、noise `0.1281/0.1531s`、delta-E `0.0023/0.0026s`、boundary `0.0477/0.0568s`、effective coverage `0.0060/0.0088s`。`source_render_cache` 两次均为 `render_hits=108`、`render_misses=9`、`quality_gate_hits=12`、`quality_gate_misses=0`、`evictions=0`、`resident_entries=9`、`resident_bytes=1,243,757,808`；`source_preflight` 两次均为 stat/hash/EXIF 各 9 次、`preflight_reuse_count=18`、`raw_decode_count=18`。

这里的 hot 仍是进程外第二次运行，因此不能期待跨进程复用 per-run LRU；它用于观察重复运行的阶段计时。当前结果的主要性能收益来自同一运行内 render→Quality Gate source 复用和 station-level ROI 将取样从 224 个 raw-owner ROI 降到 8 个 station evidence ROI，QG 取样保持在约 2.5–2.8s。后续若要让 hot 体现持久化收益，需要单独设计跨运行 source cache，不能把当前 per-run 命中计数解释成磁盘级缓存。

### 8.8 native closure 修复与 9 张 NEF 的 v12 重新验收（2026-11-02）

本阶段先修复 native closure 的误报路径，再重新跑同一连续子集 `DSC_6479.NEF`–`DSC_6487.NEF`。closure 优化在拒绝唯一冗余边后会留下 spanning tree；旧逻辑下一轮重新进入“无约束”保护分支，把刚刚发现的 direct-pair P95 失败覆盖成 `no_constraints`。现在保留原始 offending pair，并明确写入 `closure_unreliable_pair_p95`、`participating_constraints` 和 P95 值。真实冷/热报告均确认修复生效：`status=unreliable`、`fallback_reason=closure_unreliable_pair_p95`、`participating_constraints=2`、`max_direct_pair_p95_px=4.189250461725489`，offending pair 为 station `0` 的 `(row=0,column=1)` 到 station `2` 的 `(row=0,column=0)`。因此 native acceptance 仍会因直接 pair P95 超过 3px 而拒绝，但不再伪报 `no_constraints`。

为减少 Quality Gate 的误导性不可测记录，报告级 `quality_gate.unmeasurable` 现在只保留技术类不可测；内容不适用仍完整保存在各 criterion 的 `content_not_applicable_count` 和原因直方图中。该语义修复让 v12 冷/热报告的报告级不可测列表从旧报告的 141 条降为 `0` 条，技术类不可测计数同样为 `0`，但 141 个内容不适用样本并没有被伪装成可测：MTF 为 `slanted_edge_angle_out_of_range=6`、`slanted_edge_too_short=2`，noise 为 `roi_not_flat=6`，boundary 为 `boundary_low_contrast=127`。这组 141 需要后续按 criterion 选择专用 edge/flat ROI 或补充边界证据来真正减少；当前阶段先完成分类隔离，避免它们污染技术故障计数和 acceptance gate。

scale=0.5 的 boundary 采样同时使用自适应梯度门限（`max(0.15*scale, 0.06)`），并在 64px 空间单元内提前去重和过滤同 station 内部的 owner transition，减少重复 edge 重建。两次真实报告的可测 station evidence 均保持 8 个，raw-owner evidence 为 0，Quality Gate verdict 仍为 `fail`，唯一非内容不可用的失败为 `local_scale_pixel_ratio` 在 `DSC_6479.NEF` 的世界坐标 `(6735,2665)` 测得 `0.59375`（阈值 `0.99`）。

v12 Stack_Report：

- cold：`/private/tmp/raw-editor-regression/sleeping-9-owner-derived-s05-cache24-cold-v12-reports/stack-report-6bbfe4c3-7951-435b-8c6e-2afab84a1b46.json`；
- hot：`/private/tmp/raw-editor-regression/sleeping-9-owner-derived-s05-cache24-hot-v12-reports/stack-report-11f097c8-32b2-4dbb-aac5-512c60dfaf76.json`。

| 运行 | total | render | QG ROI | station evidence | technical unmeasurable | content N/A | QG ROI sampling | boundary | cache hits | QG source hits | raw decode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | 518.11 s | 504.65 s | 8 | 8 | 0 | 141 | 1.086 s | 0.041 s | 120 | 12 | 18 |
| hot | 519.95 s | 504.76 s | 8 | 8 | 0 | 141 | 1.072 s | 0.041 s | 120 | 12 | 18 |

两次运行都为 `grouping.station_count=3`、接受 2 条 station relation、连通分量为 1；SourcePreflight 均为 9 次 stat/hash/EXIF、`preflight_reuse_count=18`，受控 2.4 GB source-render cache 均为 `render_hits=108`、`render_misses=9`、`quality_gate_hits=12`、`quality_gate_misses=0`、`evictions=0`、resident entries 为 9。该验收已经完成了 closure 证据修复、技术不可测隔离、station owner evidence 和 render→Quality Gate source 复用的真实数据闭环；下一步应针对 criterion-specific ROI 继续降低 141 个内容不适用样本，并单独修复 4.189px closure 与 0.59375 local-scale failure。

## 9. 阶段 6：并发和内存控制

### 9.1 并发模型

将 pipeline 的并发拆成可观测的预算：

- RAW decode worker：默认 2–4 个，受 available memory 和 RSS 约束；
- feature worker：使用独立或受限 Rayon pool；
- pair matching worker：限制外层 pair 数，避免每个 pair 再启动过多 RANSAC 任务；
- tile/render worker：按输出 tile 大小和 resident budget 调度。

不要让“外层 pair `par_iter` + matcher 内层 Rayon + RAW decoder 内部线程”同时按满核运行。

### 9.2 运行时保护

继续使用现有 RSS sampler 和 cancellation guard，并新增：

- `max_concurrent_raw_decode`；
- `max_resident_source_pixels`；
- `max_resident_virtual_tiles`；
- `swap_risk` 记录；
- 取消时的阶段和已释放资源。

当内存接近阈值时，优先降低并发或释放 LRU；不要改变几何门禁来“救”内存。

### 9.3 通过条件

- 45/84 NEF 运行不进入持续 swap；
- peak RSS 不超过既定门槛；
- 降低 worker 数不会改变输出和关系集合；
- max concurrent decode 与 report 一致；
- 取消和错误返回都能停止 worker、释放 lease、删除临时 cache。

当前已落地的阶段 6 子项是 bounded preparation worker：大栈按可用线程和可用内存选择不超过 6 个分析 worker，并在每个 stage boundary 检查 RSS/cancellation；disconnected-component rescue 不再重新启动 RAW 解码，而是复用已准备的分析平面。`Stack_Report.resources` 仍记录全局 worker 数，下一步若要进一步拆分 RAW decode、feature、pair 和 tile worker，必须先为每一类增加实际 lease/并发计数，再调整预算，不能只把 Rayon 线程数当作 RAW 并发数。

## 10. 阶段 7：只在确认瓶颈后优化图割和 tile

只有阶段 0–6 已经证明 `tile_fuse` 或 `graph_cut_seconds` 占主要比例时，才进入本阶段。

### 10.1 低风险优化

- 单帧 station 或已确定单 owner 的 station 直接短路 graph cut；
- 复用增广路径前缀、label 候选和边界缓存；
- 先在低分辨率上求全局 owner，再只对接缝和高梯度区域做局部 refinement；
- factor=1 的 tile 避免无意义 clone 和重复面积平均；
- 共享 source analysis、tone model 和 seam plan，不能让每个 tile 独立优化。

### 10.2 硬约束

- 不允许每个 tile 使用不同的 source analysis；
- 不允许通过独立 tile 决策制造阶梯接缝或矩形曝光块；
- 不允许跳过 owner/provenance/coverage 记录；
- 每次优化都要保留输出 hash、ROI 和质量报告。

### 10.3 通过条件

- graph cut 和 tile fuse 耗时下降；
- 全图和关键 crop 没有新增双笔画、断笔、色调块、黑点或 coverage 缺口；
- 现有 graph-cut 属性测试、真实 10 张和 45/84 回归保持稳定。

## 11. 阶段 8：GPU 评估

GPU 只在以下条件同时满足后评估：

1. report 已证明瓶颈主要在可并行的像素运算，而不是 RAW 解码、读盘、候选搜索或 swap；
2. CPU 基线已经稳定，能作为逐像素对照；
3. GPU 路径有明确的颜色、alpha、owner、coverage 和边界误差容限；
4. GPU 失败或不可用时可以完整回退 CPU，且不改变来源证据。

优先考虑下采样、warp、mask、局部 tone 和 tile copy；不要一开始把整个 RAW、feature、station solve 和 Quality Gate 全部搬到 GPU。

## 12. 统一验收矩阵

每个阶段至少执行以下检查：

```bash
cargo fmt --manifest-path src-tauri/Cargo.toml -- --check
cargo check --manifest-path src-tauri/Cargo.toml --lib
cargo test --manifest-path src-tauri/Cargo.toml --lib
git diff --check
```

数据集和真实回归按当前仓库约定执行：

```bash
npm run focus-stack-datasets:check -- --require-data --hash-inputs

cargo test --manifest-path src-tauri/Cargo.toml --lib \
  panorama_stitching::acceptance_tests::real_focus_stack_fixture_from_env \
  -- --ignored --nocapture --test-threads=1

cargo test --manifest-path src-tauri/Cargo.toml --lib \
  panorama_stitching::acceptance_tests::real_ordered_panorama_full_render \
  -- --ignored --nocapture --test-threads=1
```

真实运行前检查是否已有同一 acceptance 进程；Langyuan/Wenyuan 必须串行运行。每次真实运行都要保存：

- Stack_Report；
- 2,000–3,000px whole preview；
- 左、中、右以及顶部/底部关键 crop；
- 输出尺寸、颜色/alpha、source coverage；
- 输入和输出 SHA-256；
- stage timing、pair 数、cache summary、peak RSS。

### 12.1 结果分类

| 结果 | 允许含义 |
| --- | --- |
| `passed` | 代码、报告、输出和关键 crop 均通过对应阶段门禁 |
| `diagnostic-only` | 运行完成且产物可查看，但证据或 Quality Gate 不足以作为验收 |
| `rejected` | 几何、资源、Quality Gate 或来源完整性门禁拒绝 |

“退出码为 0”“生成了 TIFF”“连接图连通”都不能单独等价于 `passed`。

## 13. 开发拆分和提交顺序

建议拆成可独立回退的提交：

1. `perf: add stack stage timing and resource counters`
2. `perf: verify singleton scan path selection and diagnostics default`
3. `perf: add run-scoped source preflight context`
4. `perf: reuse analysis proxy and feature artifacts`
5. `perf: split coarse candidate retrieval from full pair verification`
6. `perf: bound nested matching and raw decode concurrency`
7. `perf: wire virtual tile store into production station path`
8. `perf: reuse render sources in quality gate`
9. `perf: optimize graph cut and tile reuse after profiling`

每个提交都应包含：

- 影响的 report 字段；
- 是否改变默认行为；
- 冷/热运行对比；
- targeted test；
- 真实素材验证状态；
- 若未完成完整验收，明确标记为 `diagnostic-only`。

## 14. 推荐执行顺序

实际推进时按以下顺序，不要跳到 GPU 或大范围算法重写：

1. 阶段 0：加入计时、pair、cache、RSS 和 decode 并发指标。
2. 阶段 1：修正 singleton scan 的路径选择，先消除不必要的 layered cost。
3. 阶段 2：统一 SourcePreflight，减少多轮文件扫描。
4. 阶段 3：分离 fast proxy 和 full render，修复 RGB32F 重复 clone。
5. 阶段 4：coarse retrieval → top-K → 完整匹配，保留 bounded rescue。
6. 阶段 5：接入 Virtual_Tile、render 和 Quality Gate 缓存。
7. 阶段 6：限制嵌套并行和并发 RAW decode，验证 RSS 和 swap 风险。
8. 阶段 7：只有 report 证明图割/tile 是主瓶颈时才做 coarse-to-fine。
9. 阶段 8：最后再评估 GPU。

第一轮建议只完成阶段 0–2，并用兰亭 45 张 10%/1:1 做冷启动和热启动对比。只有这组报告能明确说明剩余时间主要在匹配、解码、Virtual_Tile、图割还是 Quality Gate，后续参数才值得调整。

## 15. 现有文档约束

本方案应与以下文档一起使用：

- [`focus-stack-architecture.md`](focus-stack-architecture.md)：生产路径、compositor 选择和当前真实性能基线；
- [`focus-stack-generic-regression.md`](focus-stack-generic-regression.md)：效率字段、输入变体、4–8 → 12–16 → 84 张的回归阶梯；
- [`focus-stack-reference-validation.md`](focus-stack-reference-validation.md)：真实素材的几何、来源和 Quality Gate 验收边界；
- [`focus-stack-local-work-history.md`](focus-stack-local-work-history.md)：Virtual_Tile cache、资源门槛和已完成/未完成事项的历史记录。

如果本文与这些文档中的硬门禁发生冲突，以生产代码和当前验收契约为准，并在本文件中补充变更原因和新的基线数据。

## 16. 开源项目对比结论与性能路线调整

对比 [OpenCV Stitcher](https://docs.opencv.org/5.0/main_modules/stitching.html)、[Hugin](https://hugin.sourceforge.io/docs/manual/Hugin.html)、[Enblend/Enfuse](https://hugin.sourceforge.io/docs/manual/Enfuse.html)、[focus-stack](https://github.com/PetteriAimonen/focus-stack)、[PlaneFuse](https://github.com/AdrienLF/planefuse)、[OpenFocus](https://github.com/Xinzhe99/OpenFocus) 和 [StackMFF V3](https://github.com/Xinzhe99/StackMFF-V3) 后，当前问题可以归纳为：输入模式过多但注册/融合路径还不够显式，代理图和资源预算还没有成为贯穿全流程的运行契约，单一路径的 FAST/BRIEF 与全局匹配在低纹理、重复纹理和大位移素材上存在召回与耗时的双重风险。

成熟项目普遍采用以下共同做法：先用有限尺寸的代理图完成配准和候选筛选，再执行全分辨率分块渲染；把特征、匹配、几何模型、接缝、曝光/色调和融合器拆成可替换阶段；对 focus stack 提供邻居匹配、批大小、线程数、GPU 和 CPU tiled fallback 等资源开关；同时保留多频带/小波/质量权重等经典融合器作为可对照基线。当前性能计划因此增加三条硬约束：

1. **先观察实际路径，再谈优化参数。** Stack_Report 必须同时记录配置路径、实际路径、是否使用 Virtual_Tile、singleton station 数量、孤立源数量和 fallback 原因；没有这组字段时，任何“变快/变慢”都无法解释。
2. **注册与融合必须模式化。** 书画平面扫描、摄影全景、固定机位焦点堆栈、移动源扫描分别建立 matcher/model/fusion 组合；不要让一个默认 compositor 通过隐式启发式覆盖全部素材。
3. **保留经典可解释 baseline。** 在当前 owner/graph-cut 融合之外，增加 Enfuse/多频带、complex wavelet、focus measure/深度图和 source classification 的对照接口，先用真实数据测质量、时间和内存，再决定是否替换默认路径。

### 16.1 对当前 P0–P4 顺序的影响

| 优先级 | 当前计划 | 对比后补充的验收要求 |
| --- | --- | --- |
| P0 | 路径观测、singleton scan 修复 | 报告中能解释 configured → selected 的变化；9/10 阈值有边界测试；singleton scan 不再进入高成本 layered 路径 |
| P1 | SourcePreflight、代理和缓存复用 | 同一运行内不得重复 stat/hash/decode；报告记录 proxy 尺寸、decode 次数和 cache hit/miss |
| P2 | coarse retrieval → top-K → 完整匹配 | 与 OpenCV/Hugin 风格的候选召回率、pair 数、RANSAC inlier 和耗时做对照；bounded rescue 必须保留 |
| P3 | 多 matcher/model/fusion 基线 | 至少支持 FAST/BRIEF、ORB/AKAZE 或 ECC 的可配置对照；至少保留 owner/graph-cut 与多频带/小波之一 |
| P4 | Virtual_Tile、Quality Gate 和 GPU | 继续遵守 CPU tiled fallback、来源 owner、coverage、confidence、provenance 和严格 Quality Gate；GPU 不能成为默认前置条件 |

### 16.2 必须补上的缺陷清单

- **模式识别不足：** 当前 grouping 同时承载 focus bracket 和 moving source scan，容易把单帧扫描当成多 station Virtual_Tile 任务；P0 的 singleton 观测和回退必须保持默认行为可审计。
- **单一特征路线：** FAST/BRIEF 对低纹理、重复书法和尺度变化的鲁棒性不够；P2 先做候选召回实验，P3 再引入可替换 matcher，不得直接放宽几何门禁。
- **匹配范围与成本耦合：** 全量 pair 验证会同时增加 CPU、内存和 RAW/代理读取；需要用空间邻居、粗检索、top-K 和 bounded rescue 分开计时。
- **融合器缺少对照：** 当前 owner/graph-cut 路径没有经典多频带、complex wavelet 或质量权重 baseline，出现接缝、伪影或细节损失时难以判断是注册问题还是融合问题。
- **RAW 与显示域边界不够清晰：** 线性 RAW、曝光归一化、代理图和最终显示图必须分层记录；不能通过重复 decode/clone 修复显示问题。
- **验证维度不足：** 仅看退出码、输出文件或平均耗时不够；必须同时比较 whole output、关键 crop、source coverage、owner/confidence、峰值 RSS、swap 和阶段耗时。

完整项目、算法和对照实验表见 [`focus-stack-open-source-comparison.md`](focus-stack-open-source-comparison.md)。该文档中的外部项目只作为基线和设计参考，最终默认路径仍以当前生产素材、Stack_Report 和 Quality Gate 的实测结果为准。

## 17. v14：criterion-specific ROI、closure 和 local-scale 诊断尝试（未通过）

本阶段针对连续 9 张真实 NEF（`DSC_6479.NEF`–`DSC_6487.NEF`，`scale=0.5`）中 Quality Gate 的 141 个内容不适用样本，以及 station closure 的 4.189px 直接关系误差和 0.59375 local-scale 值进行了诊断尝试。v14 未完成用户要求的修复与验收：内容拒绝重分类没有增加合格测量，local-scale 诊断化没有消除 0.59375 的真实观测，closure 的路径组合结果也不是独立直接关系证据。即使报告中的个别 verdict 变成 `pass` 或 `converged`，本阶段仍应归类为 `diagnostic-only`。

### 17.1 criterion-specific ROI 预筛选尝试

v14 在大平面且存在 station owner evidence 时，对原有共享 ROI 应用各 criterion 的内容预筛选；这尚未实现真正的 criterion-specific ROI 搜索。MTF 的候选仍全部被拒绝，noise 只有 2 个可测样本，boundary 没有可测样本，因此不能把拒绝数下降理解为证据覆盖增长。小型 fixture 保留历史共享 ROI 语义。

| criterion | 内容筛选 | Stack_Report 观测 |
| --- | --- | --- |
| `mtf50_normalized` | 输出和 reference 都必须通过 slanted-edge 预筛选；没有合格边缘的 ROI 记入 `rejected_content_count` | `candidate_count`、`eligible_count`、`selection_mode=station_evidence_shared_roi` |
| `noise_sigma_ratio` | 输出和 reference 都必须包含 flat patch；纹理 ROI 不进入 noise metric | 同上；内容拒绝不增加 `technical_unmeasurable_count` |
| `boundary_stroke_alignment` | 只保留可配对的边界候选；低对比或不可配对样本在候选阶段剔除 | `selection_mode=station_boundary_streaming_candidates`，候选总数和拒绝数分别保留 |
| `local_scale_*`、`gradient_energy_normalized`、`roi_delta_e00` | 继续使用 station owner shared ROI；完整测量失败仍按技术不可测记录 | 保持 `candidate_count`/`eligible_count` 与原始 `unmeasurable_reasons` |

新增字段的意图如下：`candidate_count` 描述进入该 criterion 的候选总数，`eligible_count` 描述通过内容契约并进入 metric 实现的数量，`rejected_content_count` 描述内容不满足契约的数量，`selection_mode` 描述候选池选择器。v14 的 boundary 计数仍存在候选数与拒绝数口径不一致，后续必须修正，不能用该值计算拒绝率。内容不适用记录从 141 减到 1 属于重分类，不代表有效测量增长；下一步必须通过扩大并搜索有效区域，实际提高 MTF、noise 和 boundary 的可测样本数。

### 17.2 local-scale 诊断化不等价于修复

v14 尝试在 `acceptance_render_scale < 1.0` 时把 `local_scale_median` 和 `local_scale_pixel_ratio` 标记为 `diagnostic=true`，仍写入测量值、timing 和 owner path，并使 native acceptance 接受无技术失败的诊断项。这只改变判定语义，没有证明 0.59375 来自降采样误差，也没有修复几何或采样问题，因此不能作为 local-scale 通过的证据。后续应保留原始测量与正式阈值，定位低比例的 world/owner 区域，并以独立全分辨率或正确归一化的 Jacobian 证据验证修复。

### 17.3 closure 路径组合存在循环验证

v14 曾在冗余直接关系触发 closure P95 超限后，沿排除该边的 spanning path 组合替代 transform，再使用它重新运行 closure solver。这样得到的近零 residual 来自路径与其自身组合的一致性，不是对原始直接关系的独立验证；即使原始 candidate diagnostics 仍保留，也不能把该结果称作 4.189px closure 修复。该循环验证方案不满足验收要求。后续应恢复独立直接关系证据，重新匹配或优化后在保留的 direct/holdout correspondences 上复核 P95；门槛保持不变。

### 17.4 v14 真实诊断运行记录

使用以下命令分别执行冷、热运行；两次运行必须使用完全相同的 9 张输入和 `RAW_EDITOR_STACK_SOURCE_RENDER_CACHE_MAX_BYTES=2400000000`：

```bash
RAW_EDITOR_FOCUS_STACK_DATA_ROOT='/Users/dp/Downloads/书画拼图' \
RAW_EDITOR_STACK_SOURCE_RENDER_CACHE_MAX_BYTES=2400000000 \
npm run focus-stack-regression -- \
  --dataset sleeping --subset sleeping-9-6479 --scale 0.5 --label v14-cold

RAW_EDITOR_FOCUS_STACK_DATA_ROOT='/Users/dp/Downloads/书画拼图' \
RAW_EDITOR_STACK_SOURCE_RENDER_CACHE_MAX_BYTES=2400000000 \
npm run focus-stack-regression -- \
  --dataset sleeping --subset sleeping-9-6479 --scale 0.5 --label v14-hot
```

冷跑报告：[stack-report-03d94727](/private/tmp/raw-editor-regression/sleeping-9-6479-s0p5-v14-cold-stack-reports/stack-report-03d94727-1fcc-4853-8afa-5bbcf886ee5c.json)；热跑报告：[stack-report-e95cb3e0](/private/tmp/raw-editor-regression/sleeping-9-6479-s0p5-v14-hot-stack-reports/stack-report-e95cb3e0-e775-4534-a1a5-495484da6f2b.json)。

| 指标 | v14 冷跑 | v14 热跑 |
| --- | --- | --- |
| 总耗时 | 552.8398s | 773.3436s |
| render 耗时 | 535.6228s | 757.8331s |
| 报告中的 QG verdict | `pass` | `pass` |
| dense ROI / station owner evidence ROI | 8 / 8 | 8 / 8 |
| MTF / noise / boundary 可测数量 | 0 / 2 / 0 | 0 / 2 / 0 |
| 报告中的 closure status / direct-pair P95 | `converged` / 约 1.02e-12px | `converged` / 约 1.02e-12px |
| ROI cache hit / miss | 0 / 8 | 0 / 8 |
| QG reference resampling | 0.8514s | 1.0253s |

两份报告中的近零 closure 均来自第 17.3 节的路径组合循环验证；QG `pass` 同时受 local-scale 诊断化和稀疏内容重分类影响，因此两次运行都没有形成正式验收通过证据。热跑期间同机还运行了另一项 `qixu` 真实任务及 Rust 编译，存在 CPU、内存竞争；上述耗时仅保留为运行观测，不能据此推断冷/热缓存收益或本轮性能变化。

v14 记录用于复现上述诊断结论，不作为 `passed` 基线。后续只有独立 closure 证据、有效 ROI 测量覆盖、local-scale 原始指标、native acceptance 和真实输出检查同时通过，才能标记为 `passed`；否则保持 `diagnostic-only` 并记录剩余门禁。

## 18. v15：独立 closure 证据、实际 ROI 搜索和输出采样密度

v15 实现用于纠正 v14 的证据缺陷。9 张 NEF 已使用同一固定构建完成冷/热运行和独立复核；closure、local-scale ratio 和 noise 有真实改善，但 MTF/boundary 仍缺有效证据，native acceptance 与 degradation 门禁尚未通过，结果保持 `diagnostic-only`。本轮没有获得热跑加速证据。

### 18.1 使用原始直接关系进行 balanced retry

移除将冗余直接关系替换为路径组合 transform 的做法。原 closure 初始化对 spanning-tree 边有较高保护权重，可能把循环误差集中压到一条真实冗余关系上；当首次求解因 direct-pair P95 超限失败时，允许一次取消 tree 额外保护的 balanced retry。重试使用同一组不可变的直接关系，不增加合成观测，也不修改匹配证据。位移、收敛、residual 和逐 pair P95 门禁保持生效。

Stack_Report 记录 `balanced_retry_attempted`、`balanced_retry_accepted` 和 `before_retry_max_direct_pair_p95_px`。重试失败时仍保留 `closure_unreliable_pair_p95` 与 offending pair，不能回落为虚假的 `no_constraints` 或用路径自洽替代独立验证。

### 18.2 criterion-specific ROI 搜索

MTF 和 noise 不再只对原来的 8 个 dense ROI 做拒绝重分类，而是在 station-owned 区域建立各自有界的候选网格，按 station 轮转遍历，避免预算只覆盖画面左上角。候选顺序由空间和内容契约决定，不根据最终 metric 是否通过重新挑选 ROI。

| criterion | v15 搜索行为 | 关键观测 |
| --- | --- | --- |
| MTF | 先搜索 256px 窗口，再搜索 512px 窗口，使 128px 连续斜边及其支撑有机会进入窗口；只有满足输出内容契约的候选才回采 Source_RAW reference | `selection_mode=station_slanted_edge_search_256_512` |
| noise | 独立搜索 128px flat patch，并校验 reference 内容契约 | `selection_mode=station_flat_patch_search_128` |
| boundary | 仅对已接受的跨 station 硬接缝搜索双侧 stroke；每个 64px world cell 最多形成一个空间独立样本，保留真实 raw owner 来源检查 | candidate、eligible、rejected-content 与 reason histogram 采用一致计数口径 |

MTF/noise 每项最多选取 24 个 reference 窗口，候选枚举和 boundary cell 也有上限。预算耗尽的未访问候选与内容拒绝、技术失败分别记录；非有限像素等技术问题不能伪装成“没有斜边”或“非平坦区域”。阈值与最低有效证据数量不变；有效样本是否增长、是否足以形成正式 verdict，须读取真实 v15 报告。

### 18.3 保留 local-scale ratio 门禁并在合成前选择输出密度

v15 保留 `local_scale_pixel_ratio` 的原始门禁：至少 99% 的测量像素要求 local scale 不低于 0.95。source 和 output 同比例缩放时，其相对 Jacobian 仍能暴露几何缩小，因此 `scale=0.5` 不再使该 ratio 自动诊断化。原有 `local_scale_median` 的 sub-scale 诊断语义仍在，不能把它误写为全分辨率通过。

输出密度在最终 compositor 分配输出画布之前选择，目标是每个最细 source 方向至少对应 1 个输出像素，参数不由 0.95 门槛倒推。实现复用 station 匹配阶段已经完成的真实 fusion：在 32px station 网格上读取 native refinement/source sampling 的 Jacobian，仅保留轻量导数证据，无额外 RAW 解码。以实际 Ownership_Map legend 和非零 owned pixels 检查源证据完整性，缺少成员、refinement 或 sampling 时明确 bypass，不能漏掉 anchor 或最高密度贡献源。

只在 world residual 为 identity 时启用统一密度 `D >= 1`，对最终 station→world 矩阵左乘 `D`；station 内部像素、native displacement、source sampling 和直接匹配关系不变。最终 compositor 由新矩阵统一建立画布、origin、tone 和 owner mapping，属于合成前的源采样规划。非 identity residual 尚未具备完整坐标适配，当前明确报告 `nonidentity_world_residual` bypass，并保留原行为。

32px 网格只是有界采样证据，不是全像素证明。报告明确写入 `sampled_native_jacobian_not_full_pixel_proof`，并保留 factor、像素倍率 `D²`、minimum/p01/median、最差 station/source/point、invalid/boundary probe 数和 bypass 原因；最终 QG 继续独立校验。密度增加必须同时满足既有 canvas 上限和内存阈值，资源不足直接拒绝，不静默降低密度。

统一输出密度也会放大输出像素域误差。原 closure 报告保留直接测量坐标和数值，`composition.output_density` 另外记录乘 `D` 后的 `direct_pair_p95_output_px`、`residual_median_output_px`、`max_corner_correction_output_px`，仍按原 3px、2px、256px 上限拒绝越界。例如 2.9px × 1.1 已超过 3px，不能因原坐标通过就允许更密输出。

### 18.4 native reference 和 ROI cache 一致性

Source_RAW reference 回采沿生产路径的 native registration/source sampling 建立，避免同一 ROI 的输出与 reference 使用不同映射。ROI cache 的运行输入 fingerprint 包含整个 ownership/coverage、world origin、source 几何、native registration/warp、实际 source sampling、residual 和已应用的 tone 参数。内部 owner 改变而 ROI 原点 owner 不变时也会失效，边界修复后的 QG 重跑不能复用旧 reference。缓存命中率不能优先于来源和映射一致性。

station reference 的完整源图缓存按 owner 批处理，常驻 map 上限为 512 MiB，跨 station 释放，单个超限源临时加载一次后释放。该上限不包含调用方当前 primary 图和临时单帧，不能用它代替进程 RSS 上限。ROI LRU 仍为 64 MiB；修复 `take→insert` 热循环重复插入 recency key 的问题，避免像素缓存有界但元数据持续增长。

定向回归覆盖：冻结真实三关系的独立 closure 重算与矛盾循环拒绝；输出密度放大后的 3px 门禁；独立斜边/平坦窗口搜索、模糊失败保留与解码失败；boundary 在实际 stroke key 上保留首次失败或技术错误；native station/single ROI 的参考采样一致性；内部 owner/refinement 变化后的缓存失效；1,000 次热命中循环的 LRU 元数据上限。最终整库测试包含这些回归，未用真实素材缺失来跳过它们。

### 18.5 v15 真实冷/热运行结果

固定输入为 `sleeping-9-6479`（`DSC_6479.NEF`–`DSC_6487.NEF`），`scale=0.5`，source render cache 上限 2,400,000,000 bytes。统一测试结果为 **935 passed / 0 failed / 40 ignored**；真实运行固定使用 release 二进制，SHA-256 为 `eff2e58d717bb0e1bc22ab2d9c99293efa5575da782d0984bc3249423359aa1e`，两次运行后复核未改变。冷跑报告：[stack-report-725a1e1a](/private/tmp/raw-editor-regression/sleeping-9-6479-s0p5-v15-cold-stack-reports/stack-report-725a1e1a-f70d-487c-a60f-67572136817b.json)；热跑报告：[stack-report-8166c534](/private/tmp/raw-editor-regression/sleeping-9-6479-s0p5-v15-hot-stack-reports/stack-report-8166c534-9c63-42a0-8f0e-63916206b756.json)。仓库持久摘要见 [`sleeping-9-v15-acceptance.json`](benchmarks/sleeping-9-v15-acceptance.json)。

这里的“冷/热”是首次与后续的两个独立进程运行，没有强制清除操作系统文件缓存。再次独立读取 9 个 NEF 后，逐文件 SHA-256 与两份报告的运行前摘要完全一致；二进制、输入和数据集范围均未发生变化。

| 指标 | v15 冷跑 | v15 热跑 |
| --- | --- | --- |
| 总耗时 / render | 625.6166s / 605.4845s | 750.5631s / 735.6875s |
| peak RSS | 9.4384 GiB | 10.2044 GiB |
| closure | `converged`，3 条独立直接关系，balanced retry 接受 | 与冷跑完全一致 |
| retry 前 / 后 direct-pair P95 | 4.189250px / 1.925860px | 4.189250px / 1.925860px |
| 输出密度 D / D² | 1.0869140625 / 1.1813821793 | 1.0869140625 / 1.1813821793 |
| D 后 direct-pair P95 / residual median / corner correction | 2.093244px / 0.468160px / 5.779306px | 2.093244px / 0.468160px / 5.779306px |
| local-scale pixel ratio | 8 个 ROI 均为 1.0，原门禁 `pass` | 8 个 ROI 均为 1.0，原门禁 `pass` |
| dense ROI / station owner evidence | 8 / 8 | 8 / 8 |
| MTF candidate / eligible / rejected | 452 / 0 / 452，`not_applicable` | 452 / 0 / 452，`not_applicable` |
| noise candidate / eligible / rejected | 37 / 24 / 13，`pass`；219 个候选因预算未访问 | 与冷跑完全一致 |
| boundary candidate / eligible / rejected | 194 / 0 / 194，`not_applicable` | 194 / 0 / 194，`not_applicable` |
| stat / hash / EXIF / RAW decode | 9 / 9 / 9 / 18 | 9 / 9 / 9 / 18 |
| QG source render cache hit / miss | 40 / 0 | 40 / 0 |
| ROI cache hit / miss | 0 / 32 | 0 / 32 |
| QG reference resampling | 0.3180s | 0.4774s |
| QG MTF 内容搜索 / boundary | 4.0974s / 0.0553s | 6.4250s / 0.1243s |
| 最终组成画布 | 7888 × 3330 | 7888 × 3330 |
| 报告级 QG / degradation result | `pass` / `degraded` | `pass` / `degraded` |

本次真实改善有独立证据：closure 直接关系 P95 从 4.189250px 降至 1.925860px，换算为更密输出后仍为 2.093244px；local-scale ratio 从先前最差 0.59375 提升为 8 个 ROI 全部 1.0；noise 可测样本从 2 个增加至 24 个，测量范围为 0.999862–1.000084。density 基于 9 个源的 100,135 个有效 Jacobian 探针，invalid probe 为 0，boundary probe 为 5,225；最差源为 station 2 / source id 2 的 station 点 `(1985,2721)`，采样最低奇异值为 0.920316。该证据仍是 32px 网格采样，不是全像素证明。

MTF 与 boundary 的证据缺口仍然存在。452 个 MTF 候选的拒绝原因为 angle out of range 253、too short 116、contrast insufficient 65、line-fit residual exceeded 18；194 个 boundary 候选全部因 low contrast 拒绝。`content_not_applicable_count=0` 是候选拒绝与 metric 失败分开计数后的字段语义，不能理解为所有内容都可测，也不能把 0 个有效样本写成质量通过。

两次 SourcePreflight 均只执行 9 次 stat/hash/EXIF，真实 RAW decode 均为 18 次；QG source cache 40 hit / 0 miss 表示回采复用了运行内源图像。两次 ROI cache 均为 32 miss / 0 hit，保留 32 项、11,010,048 bytes。冷/热各自启动独立进程，没有跨运行的内存 ROI cache，热跑时间不能归因为跨运行 ROI cache 命中。热跑总耗时与 render 耗时均高于冷跑，本轮没有热跑加速证据；同机负载与非受控的操作系统缓存仍限制性能因果判断。

两份报告的 closure、output density 和全部 criterion 记录完全一致，独立复核还确认 whole preview、6 个 crops 的 SHA-256 和 metrics 完全一致；对照记录见 [v15-comparison.json](/private/tmp/raw-editor-regression/sleeping-9-6479-s0p5-v15-comparison.json)。这证明本次固定输入和构建的结果可复现，不代表缺失的质量证据已经补齐。

**两次运行仍是 `diagnostic-only`。** `local_scale_median` 虽有 8 个数值（1.036058–1.086914），仍按 sub-scale 契约标记为 diagnostic，两次 native acceptance 都因此拒绝。degradation ledger 还包含 `grouping_accumulated_motion`、`fusion_low_sharpness_region`、`tone_gain_clamped` 三类记录，整体为 `degraded`。已检查 whole preview 和 6 个 1200px JPEG crops，并核对组成画布尺寸；这些检查不能替代原生 1:1 视觉验收。测试通过、QG 报告 `pass` 和预览产物均不等于正式验收完成。

### 18.6 后续瓶颈与推进边界

目前最明显的性能瓶颈是重复 native station fusion。v14 冷跑日志共出现 21 次 `Focus ownership solved`，这些阶段合计约 489.86s，占 render 535.62s 的约 **91.5%**；v15 冷跑仍为 21 次，合计约 551.85s，占 render 605.4845s 的约 **91.1%**。`stitch_images.rs` 的 `load_tile` 当前只有在 `source_samples.is_none()` 时才把 RGB 放入 rendered tile cache；native 路径携带真实 source sampling，因此多次 ownership、seam 和 composition 请求会重新执行 fusion。Source_RAW cache 已消除这些请求的大部分额外解码，但没有消除重复 fusion 计算。

下一阶段应优先设计有字节预算的完整 native fusion 结果复用：缓存 ownership、coverage、confidence、refinement、sampling origin 和必要像素，同时保留生产 source sampling 引用及其来源/版本信息。缓存必须把引用持有的源图像计入内存预算，并在 pose、成员或 native sampling 改变时失效；不能只缓存 RGB 后丢掉原生来源链，否则会退回二次采样并改变 QG reference 契约。需要以真实 9 张 NEF 验证 fusion 次数、耗时、RAW decode、RSS、owner/source 一致性和 QG 结果，再推广到完整素材。本文只记录该后续方案，本轮不开始实现。

质量收敛仍需继续处理 MTF 和 boundary 的 0 有效样本、sub-scale native acceptance 边界，以及 grouping/fusion/tone 的真实 degradation。应先逐类查看拒绝样本与来源，在固定阈值下验证内容搜索和测量能力；若素材确实不满足某指标的内容契约，应保留 `not_applicable` 和证据缺口，不能通过改名、清空计数或降低门槛制造正式通过。冷/热性能比较还需记录同机负载，不能把独立任务和编译竞争造成的变化解释为算法收益。

## 19. v16：受限 native fusion 结果复用

本阶段已实现第 18.6 节的重复 fusion 优化：在单次运行内复用完整 station fusion 结果，减少 matching、ownership、seam 和 composition 的重复计算。定向与整库测试、编译和数据集合同检查已经通过；同一组 9 张 NEF 的冷/热运行均已完成并独立复核。两次 fusion 都从 21 次减至 6 次，owner/source、QG 和预览结果与 v15 一致；现有全分辨率验收限制继续保留。

### 19.1 完整结果与 native source 证据一起复用

`native_fusion_cache.rs` 提供按 station 管理的字节预算 LRU，matching 的 `render_virtual_tile_match_plane` 与 composition 的 `load_tile` 共用同一个运行内 cache。条目保存 `Arc<FocusStackTileRender>`，包含 fused RGB、ownership、coverage、confidence、texture/sharpness evidence、native displacement/refinement、source sampling 和 sampling origin。命中时继续沿普通 compositor payload 构建路径处理，不把 native tile 降为只有 RGB 的二次采样结果。

源 RGB 仍由既有 SourceRenderCache 和当前 streaming payload 提供；native fusion cache 不另建一套完整源 RGB 缓存。`retained_bytes()` 按实际分配 capacity 计入 fused RGB、全部 mask/evidence plane、deformation、source refinement 和 fusion 元数据，也计入 sampling Arc 持有的 `ImageInfo` 图像、向量和 band 数据。共享引用可保守重复计费，计算本身不 clone 或展开图像；cache 层另计条目、Arc 和 LRU bookkeeping。

compositor 使用的当前 payload、为后续 QG 保留的 station 证据克隆，以及独立 SourceRenderCache 均处于 native fusion cache 预算之外。因此 cache peak bytes 不是进程内存上限，真实验收必须同时读取 RSS 峰值和既有内存拒绝门禁。进入 QG 前清空 native fusion cache，释放不再用于合成的条目；QG 使用已经建立的原生 owner/source 证据，继续执行现有标准。

### 19.2 失效契约与 registrar 状态

`native_fusion_key.rs` 将准备后的源身份与动态几何分开：source snapshot 覆盖成员顺序、source id/path、SourcePreflight 身份摘要、像素和参与融合的元数据；动态 key 覆盖 station index、station-local homographies、projection、focus warp/bands/ranges、processing fingerprint 和 native fusion 相关运行选项。source snapshot 仅在源准备及分组完成后建立，来源记录在其生命周期内保持不可变；成员或准备后的源内容变化必须重建 snapshot，几何修复必须生成新 key。

registrar 状态也是结果输入。实现对执行 fusion **之前**的完整 intra-station report 建立 SHA-256 状态摘要，包含 anchor、frame eligibility、native registration、变换及支持证据；命中要求几何 key 与该输入状态同时一致。不能只比较 anchor path，也不能用首次 fusion 新建立的执行后状态冒充执行前输入。首次建立 registrar 后，下一次请求可能因此合理地多执行一次 fusion；定向测试已覆盖首次融合发布 Failed 状态后重新执行、稳定状态命中，以及后续支持证据或 anchor 改变时失效。因此验收目标是同输入复用、变化输入重算和 owner/source 语义一致，不要求每个 station 恰好只融合一次。

状态摘要对所有 registrar 浮点字段先检查有限性；NaN 或 infinity 不会经 JSON 的 `null` 表示碰撞为相同输入，而是记录 `state_snapshot_bypasses` 并继续执行当前融合。没有 ownership masks 的 unattributed fallback 同样只返回本次结果，不进入 native fusion cache。

命中、miss 与重新执行均保留取消和 RSS 检查；融合失败不缓存，预算为零或单条结果超过预算时正常返回当前 streaming 结果并记录 bypass，不能丢失来源信息或改走弱化的 Quality Gate。所有复用只存在于当前运行，不跨进程共享旧的 native registration 或源状态。

### 19.3 预算与 Stack_Report 观测

默认预算为 **1 GiB**。`RAW_EDITOR_STACK_NATIVE_FUSION_CACHE_MAX_BYTES` 以 bytes 配置，`0` 关闭结果复用，允许范围上限为 **4 GiB**；无效或超范围配置回落到默认值。该配置独立于既有 source render cache 的预算，真实运行要同时固定两者。

`performance.native_fusion_cache` 的 `NativeFusionCacheReport` 记录以下指标，并为旧报告反序列化提供默认值：

| 指标 | 含义与验收用途 |
| --- | --- |
| `hits / misses`、`matching_* / composition_*` | 按请求阶段区分复用收益，避免把 QG 源图命中误记为 fusion 命中 |
| `fusion_count / fusion_seconds / failed_fusions` | 实际调用融合及耗时，包括失败尝试；与原有 `Focus ownership solved` 日志核对 |
| `evictions / invalidations / oversized_bypasses / unattributed_bypasses / state_snapshot_bypasses` | 区分预算淘汰、输入失效、预算禁用/超限、来源归属不足和 registrar 状态不能可靠建立摘要，不把不同 miss 原因混在一起 |
| `max_bytes / resident_bytes / peak_bytes` | 完整 retained 数据预算；清空后 resident 为零，peak 保留本次运行的最高值 |
| `resident_entries / peak_entries` | 条目数量及峰值，同时检查 LRU recency 元数据不会随热命中持续增长 |

fusion 日志中的秒数只有两位小数，累计值是近似的阶段 wall time，不能当作精确独占 CPU 时间。v16 新增计数与计时应作为主要观测，日志用于交叉检查；跨 v15 比较时需明确两者的计时口径。

### 19.4 已完成的回归验证与真实验收范围

已通过格式和编译检查、**16 项定向测试**、整库测试 **951 passed / 0 failed / 40 ignored**（249.11s），以及要求真实数据存在的数据集合同检查。两跑结束后又完成 production `cfg(not(test))` 的 `cargo check`（16.31s），格式及 diff 检查通过。定向回归覆盖完整结果命中的 Arc/owner/source/sampling 一致性、几何及完整 registrar 输入状态失效、非有限状态与 unattributed 绕过、LRU 淘汰和 retained bytes 上限、disabled/oversized 返回、失败重试不污染缓存、1,000 次热循环 recency 上限、clear 释放及旧报告兼容。source snapshot 的成员、像素、元数据与动态 key 的 projection、warp、processing 参数变化也纳入身份回归。

真实验收固定使用 release 二进制，SHA-256 为 `d883362372a6e5c6b329f9bdd96dc6d178f3cfc8dea8fa86552f32c321d396ce`。该构建用于两次运行，不能在冷/热之间重新编译或修改实现后混用结果。

真实输入固定为 `sleeping-9-6479`（`DSC_6479.NEF`–`DSC_6487.NEF`），`scale=0.5`；source render cache 为 2,400,000,000 bytes，native fusion cache 为默认 1 GiB。冷、热运行使用相同二进制 SHA，分别启动新进程，不强制清除操作系统文件缓存。已记录并核验输入 SHA、构建 SHA、总时/render、fusion count/seconds、cache 分阶段命中与预算、RAW decode 和 peak RSS；两跑完成后再次独立读取二进制和 9 个 NEF，SHA-256 与运行记录一致。

独立比较覆盖 fusion 报告（排除计时）、intra-station evidence、QG 各 criterion、closure、output density、composition、metrics、whole preview 与六张 crop；本次未自动更新基线。当前 `scale=0.5` 的 `local_scale_median` diagnostic、MTF/boundary 零有效样本、已有 degradation 与 native acceptance 拒绝均保留，没有用减少计算次数换取门禁或来源信息退化。

复跑前确认下列二进制的 SHA-256 与本节记录相同；若构建已更新，应以新版本单独记录一组对照。命令使用新的 label 保存两次独立进程结果：

```sh
cd /Users/dp/Desktop/raw-editor
shasum -a 256 /Users/dp/Desktop/raw-editor/src-tauri/target/release/deps/raw_editor_lib-814fc09cbbd3d684
for label in v16-rerun-cold v16-rerun-hot; do
  RAW_EDITOR_FOCUS_STACK_DATA_ROOT='/Users/dp/Downloads/书画拼图' \
  RAW_EDITOR_STACK_SOURCE_RENDER_CACHE_MAX_BYTES=2400000000 \
  RAW_EDITOR_STACK_NATIVE_FUSION_CACHE_MAX_BYTES=1073741824 \
  npm run focus-stack-regression -- \
    --dataset sleeping --subset sleeping-9-6479 --scale 0.5 \
    --label "$label" \
    --binary /Users/dp/Desktop/raw-editor/src-tauri/target/release/deps/raw_editor_lib-814fc09cbbd3d684
done
```

### 19.5 v15 对照与 v16 真实验收记录

v15 下表由实际报告及日志重新提取；两次均为 21 次 fusion，日志累计融合阶段分别占 render 的约 91.14% 和 91.45%。v16 冷跑报告见 [stack-report-632aec75](/private/tmp/raw-editor-regression/sleeping-9-6479-s0p5-v16-cold-stack-reports/stack-report-632aec75-6554-4d58-906a-3620a32d9009.json)，热跑报告见 [stack-report-2a006d57](/private/tmp/raw-editor-regression/sleeping-9-6479-s0p5-v16-hot-stack-reports/stack-report-2a006d57-1881-4450-9f85-e0e664b82ff5.json)。完整指标、输入/构建摘要、语义与 JPEG hash 对照保存在仓库持久摘要 [`sleeping-9-v16-acceptance.json`](benchmarks/sleeping-9-v16-acceptance.json)。

| 指标 | v15 冷跑 | v15 热跑 | v16 冷跑 | v16 热跑 |
| --- | --- | --- | --- | --- |
| fusion 日志次数 | 21 | 21 | 6 | 6 |
| fusion 日志累计耗时 | 551.85s | 672.80s | 181.25s | 208.45s |
| native cache wrapper fusion 耗时 | 未记录 | 未记录 | 190.4562s | 218.6208s |
| 总耗时 | 625.6166s | 750.5631s | 251.7920s | 281.3996s |
| render 耗时 | 605.4845s | 735.6875s | 229.9394s | 260.5933s |
| RAW decode | 18 | 18 | 18 | 18 |
| peak RSS | 9.4384 GiB | 10.2044 GiB | 8.4868 GiB | 8.7505 GiB |
| native fusion cache hit / miss | 未实现 | 未实现 | 15 / 6 | 15 / 6 |
| native fusion cache peak bytes / entries | 未实现 | 未实现 | 737,509,944 B / 3 | 737,509,944 B / 3 |
| owner/source 与 QG 对照 | 冷/热语义一致 | 与冷跑一致 | 与 v15 一致 | 与冷跑及 v15 一致 |
| native acceptance / degradation | 拒绝 / `degraded` | 拒绝 / `degraded` | 拒绝 / `degraded` | 拒绝 / `degraded` |

v16 两跑的 6 次实际 fusion 均分为 matching 3 次和 composition 3 次；composition 另外 15 次请求命中完整结果。3 次 invalidation 来自执行前 registrar 状态变化，按既有语义重新融合后才允许复用，不是无理由的缓存失效。两跑的失败、淘汰、oversized、unattributed 和 state snapshot bypass 均为 0；峰值保留 3 条、737,509,944 bytes（约 0.6869 GiB），低于默认 1 GiB，QG 前清空后的 resident bytes/entries 均为 0。

以相同口径对照 v15，本次冷/热 fusion 调用均减少 71.43%；冷跑日志累计融合耗时减少 67.16%，总耗时减少 59.75%，render 耗时减少 62.02%，峰值 RSS 减少 10.08%；热跑对应减少 69.02%、62.51%、64.58% 和 14.25%。这些是真实运行的观测，仍受同机负载和操作系统文件缓存影响，不能解释为所有素材或受控硬件下的固定加速比。新 `fusion_seconds` 冷/热为 190.456174291s / 218.620840418s，包含完整 `focus_stack_stitcher_unfilled` 调用；原日志累计 181.25s / 208.45s 是已有内部阶段计时，两者不能直接互换或混算。

v16 热跑本身比冷跑更慢（281.3996s 对 251.7920s），不构成跨进程缓存加速证据。两次独立进程各自重新建立 15 hit / 6 miss 的运行内复用，未保留上一进程的 native fusion 或 ROI cache。

两跑的 SourcePreflight 仍为 9 次 stat/hash/EXIF 和 18 次 RAW decode。source render cache 总 hit 从 v15 的 148 降为 103，其中 render hit 从 108 降为 63，是省去重复 fusion 的源请求；QG 仍为 40 hit / 0 miss，不能把请求减少误判为缓存退化。

独立复核确认 v16 冷/热的 station fusion（排除计时）、intra-station evidence、全部 QG criterion、closure、output density、composition 和 acceptance metrics 相互一致且与 v15 一致，whole preview 与六张 crop 的 SHA-256 也完全相同。`fusion_low_sharpness_region` 记录从 21 条变为 6 条只是减少重复计算后的日志数量变化；该原因、其余 degradation 原因以及整体 `degraded` 均保留，不构成质量改善。MTF/boundary 仍为 0 个有效样本及 `not_applicable`，`local_scale_median` 仍为 sub-scale diagnostic，native acceptance 继续拒绝；本次两跑只能作为 `scale=0.5` 的性能与结果一致性证据。

本次视觉产物范围为 preview-only：核对了 whole preview 和 6 张 JPEG crop，没有完成 native TIFF 全像素或原生 1:1 视觉验收。语义摘要和预览 hash 相等支持本轮回归结论，不能替代缺失的全分辨率证据。v15 对照仍见仓库持久摘要 [`sleeping-9-v15-acceptance.json`](benchmarks/sleeping-9-v15-acceptance.json)；上述证据共同支持本阶段减少重复 fusion、保留现有来源和门禁语义的实现结果。


## 20. v17：同一组 9 张 NEF 的原生 TIFF 冷/热验收

本阶段将第 19 节的 `scale=0.5` 诊断运行推进到 `scale=1`，复用同一份已验证 release 构建，导出完整 TIFF，并对 TIFF 本身执行原生 1:1 关键区域检查。正式 `measured-cold` 与 `measured-hot` 均已完成；此前两次测量准备的产物和失败原因分别保留。两次 native harness 和 QG 均通过，TIFF 文件逐字节一致，但独立 IFD 检查发现缺失 `ExtraSamples` alpha 声明，`artifact_contract=fail`、`native_acceptance_complete=false`。来源排除、内容不适用指标及 `degraded` 也继续保留，本阶段不能笼统宣布完整原生验收通过。1:1 裁剪的逐项视觉结论单独记录。本阶段不调整 owner/source 契约、Quality Gate 阈值或 RSS 门禁。

### 20.1 固定构建、输入与原生运行配置

输入继续为 `sleeping-9-6479`，严格选取 `DSC_6479.NEF`–`DSC_6487.NEF`。复用 v16 release 二进制 `/Users/dp/Desktop/raw-editor/src-tauri/target/release/deps/raw_editor_lib-814fc09cbbd3d684`，SHA-256 为 `d883362372a6e5c6b329f9bdd96dc6d178f3cfc8dea8fa86552f32c321d396ce`。v16 的 16 项定向、951 项整库通过和 40 项 ignored 是该构建已有的验证记录，不冒充本阶段重新执行的测试。冷热之间不得修改或重新构建二进制；每次执行前后均复核构建 SHA，结束后独立复核 9 个源文件的 SHA。

| 配置 | v17 固定值 | 与 v16 的关系 |
| --- | --- | --- |
| source subset / alignment / blend | `sleeping-9-6479` / `auto` / `focus-stack` | 保持同一输入与路径 |
| 请求 scale / output | `1` / `--write-output` | 从缩比例 preview-only 改为原生 TIFF |
| coverage alpha | `RAW_EDITOR_STACK_ACCEPTANCE_COVERAGE_ALPHA=1` | 使用生产 coverage alpha，检查未覆盖区，不以黑边代替透明覆盖 |
| SourceRenderCache | 2,400,000,000 bytes | 与 v16 相同 |
| NativeFusionCache | 4,294,967,296 bytes（4 GiB） | 从 v16 的 1 GiB 提高到现有配置上限 |
| stitch/QG RSS 门禁 | 24 GiB，机器物理内存 32 GiB | 既有门禁保持不变；TIFF 导出另取 OS 全进程峰值 |
| run labels | `v17-native-measured-cold` / `v17-native-measured-hot` | 各自独立进程，串行执行 |

按像素面积约为四倍进行容量估计，v16 三个 fusion 条目的 737,509,944 bytes 在原生尺度可能约为 2.95 GB；这只是配置依据，实际驻留量以报告为准。4 GiB 是已有受限缓存的上限，没有扩大进程 RSS 门禁。完整源 RGB 的四倍估计约 4.98 GB，仍可能超过 2.4 GB 源缓存，因而不能预先承诺 RAW decode 继续为 18 次。需要记录 evictions、oversized bypass、decode 与 RSS 的实际变化。

scale 和 native cache 预算都发生变化，因此 v16 与 v17 不是同分辨率、同预算的 A/B 性能实验。v17 最终冷/热之间固定所有参数，不强制清除操作系统文件缓存，也没有跨进程内存 cache；“热跑”只是后续的新进程运行。正式对照前的两次测量准备分别保留，不混入最终冷/热统计：

- `v17-native-cold`：`interrupted_instrumentation_precheck`，SIGINT 结束码 130。为补齐 TIFF 导出阶段的 OS 进程峰值观测而中断，不作为验收运行。
- `v17-native-full-cold`：`instrumentation_failed_reference`。已生成 `15701×6585` 原生 TIFF，Rust native harness 记录 `1 passed / 0 failed`；但 `/usr/bin/time -l` 在结尾调用 `sysctl kern.clockrate` 被沙箱拒绝，报 `Operation not permitted`，没有输出 maximum RSS，并使 wrapper/runner 结束码为 1。这是计量失败，不能改写为算法或 native harness 失败，也不能用其缺失的 OS 峰值完成内存验收。

两次准备运行已经读取过源数据，因此 `measured-cold` 表示首个使用最终计量方法的完整对照运行，不等于清空 OS 缓存后的磁盘冷启动。`full-cold` 的 TIFF、日志和报告保留用于来源核验及问题追溯，最终性能和视觉结论使用 `measured-cold` / `measured-hot` 这对记录。

实际执行脚本为 [`run-native-tiff-v17.py`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v17/run-native-tiff-v17.py)，它先删除继承环境中的全部 `RAW_EDITOR_*` 变量，再只设置数据根、两项缓存预算、coverage alpha、`auto` 和 `focus-stack`。这样不会意外继承 `PREVIEW_ONLY`、geometry preview、alignment-only 或禁用 native refinement 的诊断开关。正式运行改用 [`native-tiff-rusage-binary-v17.py`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v17/native-tiff-rusage-binary-v17.py)，由 Python 只启动一个上述固定 Rust 子进程；退出后用 `resource.getrusage(resource.RUSAGE_CHILDREN)` 取得该子进程的资源统计，并打印 `NATIVE_PROCESS_USAGE` JSON。Darwin 的 `ru_maxrss` 按 bytes 记录，JSON 同时包含全程 wall/user/sys 和原始 `child_exit_code`，避免依赖受限的 `sysctl`。这一路径已用单子进程分配并触碰 32 MiB 的探针验证，实测 maximum RSS 为 48,267,264 bytes。两跑固定相同二进制与 wrapper 配置；脚本核对二进制 SHA、记录 wrapper SHA，拒绝覆盖同 label 的历史日志，并分别写入 `raw-editor-v17-native-measured-cold-execution.json` 与 `raw-editor-v17-native-measured-hot-execution.json` 执行清单：

```sh
python3 /private/tmp/run-native-tiff-v17.py cold
python3 /private/tmp/run-native-tiff-v17.py hot
```

对应 harness 命令为 `npm run focus-stack-regression -- --dataset sleeping --subset sleeping-9-6479 --scale 1 --write-output --label v17-native-measured-cold --binary /private/tmp/native-tiff-rusage-binary-v17.py`，热跑只改 label。复跑时应使用新 label，并保留相同的环境清理和 SHA 检查；不能直接覆盖本次记录。

### 20.2 原生输出与门禁的独立核验

请求参数 `--scale 1` 不能单独证明实际原生输出。需要检查报告的实际 render scale、准备后源尺寸和最终画布，确认路径为 `layered_virtual_tile`，未切换到 panorama、代表帧预览或其他 fallback。`local_scale_median` 必须按原生契约给出实际 verdict，不能继续以 sub-scale diagnostic 解释；closure 和输出密度增加后的误差也按既有原生像素上限判断。

每次独立读取 Stack_Report 中的 station membership、accepted relations、closure、intra-station frame records、fusion ownership、source provenance、coverage、native sampling/refinement、output density、各 Quality Gate criterion、tone 和 degradation。有效 ROI 数、station owner evidence、candidate/eligible/rejected-content/technical-unmeasurable 计数及拒绝原因均保留。MTF、noise 或 boundary 没有适用内容时如实记为 N/A，技术失败仍是失败；不把缩比例的旧 verdict 搬到原生运行，也不修改门槛让本次通过。

harness 会在写出产物后验证 native acceptance，而 Node runner 可能把“已输出但 native 拒绝”记录为 notice 或 `no-baseline` 并返回 0。因此须同时读取原生 harness 的失败原因和报告，不能只看外层 exit code、TIFF 存在或 QG 总体字段。若几何、内存或导出阶段提前拒绝，记录拒绝阶段和缺失产物；不降到 scale < 1，也不把未生成 TIFF 的运行记为已完成原生视觉验收。

性能核验区分三个计时口径：报告的 pipeline total/render、`native_fusion_cache.fusion_seconds` 的完整融合调用计时，以及执行清单的进程 wall time。原有 `Focus ownership solved` 日志只保留两位小数，作为内部阶段累计对照。已核实 TIFF/JPEG 编码及导出处于报告 RSS 采样生命周期之外。因此正式两跑额外采用单 Rust 子进程的 Darwin `getrusage(RUSAGE_CHILDREN)`，取得包含导出的 maximum RSS 和完整 wall/user/sys；该值与 Stack_Report 的 stitch/QG 采样峰值分列，不混用，也不把外层 Node 或 Python 父进程的 RSS 当作 renderer 峰值。计量 marker 缺失、Rust `child_exit_code`、wrapper 结束码与 runner 结束码分别记录，以区分算法失败、测量失败和外层汇总判定。报告原有 24 GiB 取消门禁的覆盖范围仍是 stitch/QG；新增 OS 计量不会扩大或替换这项门禁。

### 20.3 TIFF 元数据与原生 1:1 视觉检查

对真实 TIFF 核验文件 SHA、尺寸、样本位深、通道/alpha、ICC 和可解码性。coverage alpha 应与未覆盖区域保持一致；JPEG preview 的背景颜色不作为 coverage 证据。冷热 TIFF 若文件 SHA 不同，进一步区分容器元数据差异与解码后像素差异，不仅比较缩略图。

现有 harness 的六张 JPEG crop 会把约三分之一宽、二分之一高的区域缩至 1200px，不能作为原生 1:1 证据。本阶段另从 TIFF 按原生整数坐标直接裁切 PNG，不使用 resize；记录每张裁剪的 `(x, y, width, height)`、导出尺寸和 PNG 文件 SHA。3 个 coverage 窗口另在 `alpha-audit.json` 保留 decoded RGBA16 SHA；没有将 PNG 文件摘要冒充所有窗口或全 TIFF 的解码像素摘要。裁剪方式使用整数坐标、16-bit PNG header 校验和合成 RGBA16 保真试验验证；没有逐像素比较全部实际 TIFF 裁片与源 TIFF，不能声称这项实图逐像素对照已经通过。坐标以本次原生报告与 TIFF 为准，不能机械地把 `scale=0.5` 坐标乘二，也不能把 station-local 或 world 点直接当 canvas 点。

检查包括整图概览、上/下与左/中/右六个分区，并补充实际跨 station 接缝、细笔画/轮廓、焦点交接、低清晰区和 coverage 边界。逐处记录模糊、重影、断线、矩形曝光块、接缝色差、透明孔洞或残留等观察；没有可疑现象只代表已检查窗口，不等于人工查看了每一个 TIFF 像素。独立视觉检查与 Quality Gate 同时保留，不能相互替代。

裁剪和解码验证均在两次计时运行完成后进行，避免额外 TIFF 解码、图片工具或编译影响冷/热的耗时及峰值内存。31 个原生窗口及其坐标、尺寸和 SHA 已记录在 [`visual/manifest.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v17/visual/manifest.json)，逐项视觉观察与覆盖限制见第 20.7 节。

### 20.4 原生冷/热结果记录

| 指标 | v17 原生冷跑 | v17 原生热跑 |
| --- | --- | --- |
| 实际 render scale / 准备后源尺寸 / selected path | `scale=1`，9 源准备后均为 8312×5542；`layered_virtual_tile` | 与冷跑一致 |
| TIFF 尺寸 / bit depth / samples / ICC | 15701×6585 / 16-bit / 4 samples / 480 B ICC | 与冷跑一致 |
| TIFF alpha 声明 / artifact contract | 缺少 ExtraSamples tag 338 / `fail` | 与冷跑一致 |
| fusion 次数 / wrapper 耗时 | 6 / 175.849115s | 6 / 211.486998s |
| fusion 日志累计耗时 | 158.00s | 188.87s |
| pipeline total / render | 480.916285s / 466.884727s | 404.501396s / 378.317897s |
| Rust 子进程 wall time（含导出） | 484.780833s | 407.503396s |
| 外层执行清单 wall time | 485.449634s | 407.770942s |
| RAW decode / SourceRenderCache hit-miss | 70 / 62–61 | 70 / 62–61 |
| NativeFusionCache hit-miss / evictions / bypass | 15–6 / 0 / 0；状态失效 3 次 | 与冷跑一致 |
| NativeFusionCache peak bytes / entries | 2,944,716,552 B / 3（约 2.7425 GiB） | 与冷跑一致 |
| Stack_Report peak RSS（stitch/QG） | 11,223,105,536 B（10.4523 GiB） | 10,707,648,512 B（9.9723 GiB） |
| OS maximum RSS（Rust 进程，含导出） | 11,256,299,520 B（10.4832 GiB） | 11,283,611,648 B（10.5087 GiB） |
| owner/source 与 coverage 证据 | 8 个 dense ROI；density 记录 8 个有效源；`DSC_6484.NEF` 被原生门禁排除 | 与冷跑语义一致 |
| QG / native harness / degradation | `pass` / Rust `1 passed` / `degraded` | 与冷跑一致 |
| TIFF 文件对照 | 827,135,978 B；两份文件 SHA 完全一致 | 与冷跑相同 |
| 1:1 关键区域视觉检查 | 31 个原生窗口逐项检查，带限制完成；主代理另复核其中 13 个 | 同 SHA TIFF，复用同坐标检查 |

两跑的 Rust 子进程和外层 runner 均结束为 0，固定二进制 SHA 前后不变，9 个原始 NEF 的 SHA 也经独立重读核对。产物已归档至 `output/acceptance/sleeping-9-native-v17/`：[冷跑 TIFF](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v17/cold.tiff)、[热跑 TIFF](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v17/hot.tiff)、[冷跑 Stack_Report](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v17/cold.stack-report.json)、[热跑 Stack_Report](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v17/hot.stack-report.json)以及同目录下的 execution、log、metrics 和 `visual/`。持久验收摘要为 [`sleeping-9-v17-native-acceptance.json`](benchmarks/sleeping-9-v17-native-acceptance.json)。

两份 TIFF 均为 827,135,978 bytes，SHA-256 同为 `e1cbb43dbf6d7550a74a3bde756e8a6dde2432ee7388aac1a4cb3ee93b635d94`；application JPEG、whole preview、六张既有 crop 和 metrics 的摘要也相同。两跑 grouping、intra-station、virtual tiles、station relations、closure、residual warp、tone、composition、fusion（排除计时）和 QG 语义完全一致。文件相同证明这组输入和构建的结果可复现，不能修复两份文件共同缺失的 alpha 声明。

热跑全 Rust 进程 wall time 比冷跑少约 15.94%，pipeline total 少约 15.89%；但两跑融合次数同为 6，RAW decode 同为 70，native/source cache 计数一致，且热跑 fusion wrapper 耗时反而增加约 20.27%。两次均为独立进程，冷跑前也已有计量准备运行，不能将 wall time 缩短归因于融合次数减少或跨进程缓存命中。OS 全进程 RSS 热跑略高约 0.24%，也说明不能用仅 stitch/QG 的采样峰值代替导出在内的峰值结论。

### 20.5 原生运行的新发现与推进边界

**原生局部尺度门禁已有实际测量通过，但内容证据和 degradation 仍需单列。** 两跑的 `local_scale_median` 为 1.028809–1.082031，`local_scale_pixel_ratio` 的 8 个 ROI 均为 1.0，两项 `diagnostic=false` 并通过原生门禁。closure 的 direct-pair P95 为 2.252797px，密度因子 `D=1.08203125` 后为 2.437597px；tone boundary Delta E00 最大 0.064125，低于原 1.5 上限。

MTF 搜索 512 个候选但 0 个有效样本，仍为 N/A；boundary 搜索 388 个候选，全部 low contrast，仍为 N/A。noise 的 43 个候选中 24 个有效、19 个内容拒绝，测量范围 0.996731–1.000001，结果通过。QG 总体 `pass` 和 native harness `1 passed` 是其现有契约的真实结果，不能扩写成 MTF/boundary 已测量通过或整幅原生图像没有缺陷。

**9 个输入不等于 9 个源都贡献原生像素。** 正式两跑确认 `DSC_6484.NEF` 的空间支持为 0.085036，低于原 0.20 门禁，状态 `failed`；v16 的 scale=0.5 对应值为 0.806398、状态 `local`。原生 density 证据保留 8 个有效源。这是跨尺度配准/支持证据差异，不能为保留第九帧而放宽门槛，也不能将源排除隐藏在“全部输入已读取”之后。后续需定位坐标尺度、native patch 支撑与成员排除的形成过程，并结合最终覆盖及原生局部图像判断实际影响；本阶段先保留真实证据，不改算法。

两跑 ledger 均为 `degraded`，含 `grouping_accumulated_motion` 4 条、`intra_station_registration_failed` 3 条、`intra_station_rejected_from_group` 2 条和 `fusion_low_sharpness_region` 6 条。native verifier 通过与该 ledger 状态并列记录：一个说明当前几何/QG/tone 验证条件满足，另一个保留上游成员和清晰度的降级事实，不能相互覆盖。

**新的性能瓶颈是原生源缓存容量，不是重复 stat/hash/EXIF 或 native fusion 失效。** SourcePreflight 仍各执行 9 次 stat/hash/EXIF；70 次 RAW decode 可核对为 9 次分析准备加 61 次 SourceRenderCache miss，其中 render 为 49 miss、QG 为 12 miss。源缓存另有 render 16 hit、QG 46 hit，共 62 hit；发生 57 次淘汰，峰值/最终驻留为 2,211,124,992 bytes、4 个源。每个准备后的 8312×5542 RGB32F 源约 552,781,248 bytes，8 个有效源约 4.42 GB，超过当前 2.4 GB 源缓存。作为容量事实，即使使用现有 4 GiB 源缓存上限，也略小于这 8 个完整 RGB32F 源的总量；后续需基于源请求顺序、保留周期及共享/分块策略减少重解码，不能只用提高 cache hit 比例或放宽内存门禁宣称解决。

native fusion 复用仍有效：6 次执行分为 matching 3 次和 composition 3 次，另 15 次 composition 请求命中；仅有预期的 3 次 registrar 输入状态失效，无失败、淘汰或 bypass，QG 前清空后 native cache resident 为 0。因此后续优化应分别计量 RAW 重解码与 fusion 计算，不把两类成本合为“融合缓存失效”。

当前 Stack_Report 在 stitch/QG 结束时写出，其 `output.written=false`、空路径等默认值不会在 harness 随后的 TIFF 导出后回填；这也是为何导出完成与位深、alpha、ICC 必须以实际 TIFF、harness 日志和全进程计量独立核验。此次独立 IFD 检查已经检出报告和 native verifier 未覆盖的实际导出缺陷。

### 20.6 P0 导出缺陷：缺少 TIFF ExtraSamples alpha 声明

独立读取两份 TIFF 的 IFD 后确认：尺寸 15701×6585、`BitsPerSample=[16,16,16,16]`、`SamplesPerPixel=4`、ICC profile 为 480 bytes，但 **没有 ExtraSamples tag 338**。四通道像素中存在 coverage alpha 数据，不等于文件已经声明第四通道的 alpha 语义。当前 canonical RGB 未按 alpha 预乘，产物应明确声明一个 unassociated alpha sample，即 `ExtraSamples=[2]`；解码器自行猜测第四通道为 alpha 或 `tiffinfo` 的推断不能替代这个 IFD 契约。

问题定位在 [`output_encoding.rs:70`](/Users/dp/Desktop/raw-editor/src-tauri/src/image_stack/output_encoding.rs:70) 的 TIFF 分支：当前通过 `image::codecs::tiff::TiffEncoder` 写出 RGBA16，并保留 ICC，但这条本机构建依赖路径没有写入所需的 ExtraSamples 声明。因此两跑 `artifact_contract=fail`，`native_acceptance_complete=false`。这是实际 writer 导出缺陷，并非 alpha 像素缺失；两份 TIFF 的 SHA 相同、native harness 通过和 QG 通过都不能消除该问题。本阶段保留原始二进制和 TIFF，不修改 writer，也不对已计量产物打补丁后冒充原始输出。

下一阶段 P0 应在 TIFF 写出分支直接使用 `tiff` encoder，以 `RGB16`（8-bit 对应 `RGB8`）作为三个颜色样本的基础，再调用 `extra_samples(&[ExtraSamples::UnassociatedAlpha])` 增加一个 alpha 样本，并写入现有四通道数据。**不能使用 `RGBA16` 后再添加一个 extra sample**，否则会被配置成五个样本。修复必须保留 ICC、既有 RGB/coverage alpha 像素值和 owner/source 结果；RGB-only TIFF 的三样本行为也要保持。

定向验收应读取真实 IFD，要求 tag 338 为 `[2]`、SamplesPerPixel 为 4、位深和 ICC 正确，并比较修复前后解码的 RGB 与 alpha 数据；覆盖 16-bit、8-bit、RGB-only、透明边界与不透明内部。完成 writer 回归后，再生成新的原生 TIFF 并验证其 metadata、像素和跨解码器 alpha 解释，不能只依赖当前对输出像素的推断式解码测试。

### 20.7 原生 1:1 视觉检查记录

原生检查以归档的 `cold.tiff` 为唯一裁剪源；两份 TIFF 的 SHA 完全一致，因此同一套原生窗口也代表热跑的相同像素。按照 [`manifest.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v17/visual/manifest.json) 的整数坐标直接裁切，没有 resize。PNG 保留 16-bit 样本，查看器的屏幕呈现可能为 8-bit；整图概览经过缩小，仅用于定位，不计作 1:1 证据。

| 窗口类别 | 数量 / 原生尺寸 | 覆盖范围 |
| --- | --- | --- |
| 六分区中心 | 6 / 1024×1024 | 上下两排、左中右各区 |
| 内容目标 | 10 / 768×768 | 左右印章、栏杆连接、中央栏杆、榻足及左右鞋细节 |
| coverage 边界 | 3 / 768×768 | 顶边、底边阶梯及右边界 |
| 原生 QG ROI 上下文 | 8 / 768×768 | 报告中全部 8 个 dense ROI 所在位置 |
| 报告跨 station 边界上下文 | 4 / 768×768 | 报告可用 boundary 样例中的分散窗口 |

最初名为 `shoe-fine-lines` 的窗口实际落在两鞋之间的榻足，不能用它声明已检查鞋部。因此另补 `left-shoe-detail` `(7906, 3745, 768, 768)` 和 `right-shoe-detail` `(9869, 3560, 768, 768)` 两个真正覆盖左右鞋的窗口，最终共 31 个。坐标和每张 PNG 的 SHA 以 manifest 为准；world origin 为 `(-106, 4)`，已经完成 world→canvas 坐标换算。报告只保存有限前缀的跨 station boundary 样例，上表 4 个窗口不代表所有接缝；fusion `low_sharpness_regions` 缺少最终 station→world 映射，未将这些局部坐标误当作 canvas 坐标，也不能宣称逐一检查了所有低清晰区。

三个 coverage 窗口另做了 [`alpha-audit.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v17/visual/alpha-audit.json) 的第四样本数值检查：底边、右边和顶边分别有 219,203、60,850、154,740 个透明像素；三个窗口均为二值 coverage，部分透明像素数和不透明纯黑像素数均为 0。这区分了当前样本里的未覆盖区域与已覆盖纯黑像素，但不会补上原 TIFF 缺失的 alpha 声明，也不证明整幅画布没有内部覆盖缺口。

主代理另作了 [13 个原生窗口及概览的交叉复核](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v17/visual/root-cross-check.json)，覆盖六分区、榻足、中央栏杆、右栏杆接点、左侧小印章、底边 coverage 和新增左右鞋。所查栏杆与鞋的轮廓连续，未见明确的大范围重复轮廓、断裂或矩形曝光块；基底织纹可见，纸边红印相对柔软。由于没有匹配同坐标的源 RAW 最佳焦面，不能将软边归因于融合、拍摄焦点或原始印迹，也不能据此宣称处处清晰。

[31 个窗口的逐项视觉记录](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v17/visual/visual-review.md)（[JSON](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v17/visual/visual-review.json)）包含新增左右鞋及最后 4 个 seam 样例的实际查看结果：所查鞋轮廓与丝织纹理保持连续，没有明确的大幅重复轮廓或断线。有限的 seam 样例主要落在左侧纸边与印章附近，可见红印软边及竖向纸折痕；这些外观未经源 RAW 同位置对照，不能归因为融合缺陷。顶、底、右侧 coverage 的阶梯形外轮廓继续保留，第四样本数值检查只说明这些已查边界对应未覆盖区，不意味着输出已经成为没有透明边缘的完整矩形画幅。

本次真实 TIFF 的原生 1:1 抽样视觉检查及独立交叉复核已带限制完成（`completed_with_limitations`），视觉记录仍为 `overall_native_visual_acceptance=not_fully_accepted`，证据覆盖范围止于上述窗口。8 个 QG 窗口是原始 ROI 的较大上下文，部分上下文包含 coverage 外边缘，不能据此反推原始较小 ROI 的测量无效。它不能替代所有接缝、全部低清晰区或全图每个像素的检查，不能将 MTF/boundary 的 N/A 变成已测量通过，也不能清除 source 排除和 `degraded`。完整原生验收仍因第 20.6 节的 TIFF alpha 声明缺失而为 `false`；下一阶段先修复并验证 writer，再继续源缓存重解码和原生支持证据问题。

## 21. v18：P0 TIFF 声明与导出验收闭环

本阶段先处理第 20 节已确认的 TIFF `ExtraSamples` 缺失，以及“文件已写出但导出契约未进入最终验收”的缺口。实现覆盖实际生产 TIFF writer、通用编码 helper、导出后的 IFD 校验、Stack_Report 回填和 Node runner 失败传播。P0 导出阶段现已完成：最终构建的代码检查、963 项整库测试、真实 9 张 NEF 单次原生导出及 14 项独立审计均通过，`export_phase=pass`、`artifact_contract=pass`。首跑曾检出 571 处报告数字低位漂移，已改为只回写 output、保留原始证据片段，并用新构建重新验收；首跑继续保留为问题参考。最终 TIFF 全 RGBA 样本、ICC 和 10 项几何/质量语义证据与 v17 精确一致。完整原生质量验收仍为 `false`：第九帧排除、degraded、MTF/boundary 的 N/A 和视觉限制没有被本次导出修复消除。owner/source、配准门槛、Quality Gate 阈值和 RSS 上限保持原值，原始 v17 TIFF 不修改。

### 21.1 对第 20.6 节调用链定位的更正

第 20.6 节确认“v17 TIFF 缺少 tag 338”的文件证据成立，但把本机真实 fixture 的直接编码路径归到 `output_encoding.rs` 中的通用 `image::TiffEncoder` 分支不够准确。重新沿实际调用链核对后，桌面 fixture 的 `write_srgb_tiff` 调用 `write_image_stack_output`，经 `encode_image_stack_file` 进入 [`export_processing/tiff_writers.rs`](/Users/dp/Desktop/raw-editor/src-tauri/src/export_processing/tiff_writers.rs) 的生产 writer。这里的标准 TIFF 与 BigTIFF RGBA 分支都直接以 `RGBA16/RGBA8` 建立图像，没有显式声明 extra sample，因而同样受影响。

[`image_stack/output_encoding.rs`](/Users/dp/Desktop/raw-editor/src-tauri/src/image_stack/output_encoding.rs) 中的通用 `Write + Seek` TIFF helper 是另一条受影响路径，不能只修该 helper 就宣称真实 fixture 与桌面生产导出已经修复。本节修正定位，保留第 20 节作为当时的验收记录；修复范围同时覆盖生产标准 TIFF、生产 BigTIFF 和通用 helper。

### 21.2 Writer 改动与像素保持要求

有 alpha 的 8-bit / 16-bit TIFF 统一以 `RGB8/RGB16` 为基础颜色样本，随后显式调用 `extra_samples(&[ExtraSamples::UnassociatedAlpha])`。实际四通道数据按原顺序写入，目标 IFD 为 `SamplesPerPixel=4`、`ExtraSamples=[2]`。不在 `RGBA8/RGBA16` 后再增加 extra sample，避免写成五样本文件。RGB-only 仍为三个颜色样本且不写 tag 338。

标准 TIFF 与 BigTIFF 的预期输入长度按 `RGB 样本数 + alpha 样本数` 检查，维持错误长度拒绝、原 ICC 开关及现有 metadata 路径。通用 helper 直接使用已有 `tiff` 依赖，保留原位深转换和 sRGB profile 行为，不增加新依赖。修复只改变容器的明确声明，不允许改变已经确定的 RGB 或 coverage alpha 值；因此 v17/v18 文件 SHA 可以因 IFD 改变而不同，像素是否保持必须以解码数据对照判断。

定向测试覆盖标准 TIFF / BigTIFF、8-bit / 16-bit、透明 / 部分透明 / 不透明样本、ICC 开启与关闭、RGB-only，以及错误样本数。真实 IFD 的 tag 338、样本数与位深检查和解码后逐样本对照分别保留，不能只用解码器“猜出 RGBA”作为格式正确的依据。

### 21.3 导出门禁和 Stack_Report 生命周期

[`output_validation.rs`](/Users/dp/Desktop/raw-editor/src-tauri/src/image_stack/output_validation.rs) 在发布临时文件前直接读取 TIFF IFD：alpha 输出必须有四个样本及 `ExtraSamples=[2]`；RGB-only 必须有三个样本且没有 ExtraSamples。缺失、错误或矛盾声明一律失败。随后继续核对实际尺寸、颜色类型、位深和所要求的 sRGB profile。IFD 检查只读取头部，不为验收分配完整图像或触发 RAW 解码。

真实 fixture 改用 `write_srgb_tiff_for_acceptance`，共享生产写出与校验边界，并在导出后更新原 Stack_Report 的 `output`：实际路径、尺寸、格式、位深、ICC、`alpha_preserved`、`tiff_extra_samples`、`artifact_validation` 和错误信息。新字段带默认值，旧报告保持 `not_run`，不能被当作已通过新的产物校验。报告更新不修改 QG、fusion、owner/source 或已有 degradation 证据。

导出或格式校验失败时，fixture 记录 `artifact_validation=fail` 并返回失败；报告回写失败也使本次验收失败，避免产物和报告脱节。成功导出后，既有 native geometry / QG / tone 验证继续运行。这里的 `output.artifact_validation=pass` 只说明 TIFF 导出契约通过，不代表所有质量问题、来源排除或视觉限制已解决。preview-only 运行不执行 TIFF 导出，维持独立的未运行状态。

### 21.4 Runner 的失败传播与旧产物处理

[`tests/focus-stack-regression.mjs`](/Users/dp/Desktop/raw-editor/tests/focus-stack-regression.mjs) 将 native harness 拒绝作为失败，不再仅作为 notice。非 preview-only 运行同时要求非空 TIFF、匹配请求路径的已验证 `output`、与渲染结果相同的尺寸、16-bit 和预期 ICC；启用 coverage alpha 时还必须有已核验的 `[2]` 声明。runner 汇总保留 `artifactContract`，从而区分 `pass`、`fail` 与 `not-run-preview`。

旧基线或 `--reuse-artifacts` 不得通过重读旧日志自动取得新的导出通过状态；未写入已验证产物字段的历史记录，需要明确拒绝或重新执行。当前 v17 文件继续保留其失败记录，不对旧 TIFF 补标签后覆盖原始 SHA 和计时证据。

### 21.5 最终验证记录与真实 9 张 NEF 单跑

正式验收继续使用 `DSC_6479.NEF`–`DSC_6487.NEF`、`scale=1`、coverage alpha、2,400,000,000 B source cache 和 4 GiB native fusion cache。最终固定 release 二进制为 `/private/tmp/raw-editor-v18-verified-binary`，SHA-256 为 `7263f27b9a9395dc7cc405602b36734e455b14d564b47dd5cd6bb233589adff9`；构建、执行及独立磁盘读取的 SHA 一致，9 个源文件也经磁盘重读验证。正式结果归档在 `output/acceptance/sleeping-9-native-v18-export/verified/`，与此前的首跑参考分开。

| 验证项 | 最终结果 | 通过依据 |
| --- | --- | --- |
| writer / 导出门禁 / 报告定向测试 | TIFF 定向 19 passed / 0 failed；补充报告回写 2 passed / 0 failed | 标准 TIFF / BigTIFF、8/16-bit、ICC、alpha/RGB-only、像素 round-trip、错误声明拒绝、原始证据和未知字段保持 |
| runner 失败传播测试 | 8 passed / 0 failed | harness 失败、未验证导出、旧基线及 preview-only 各自状态明确 |
| 数据集契约 | `--require-data` 通过 | 对实际素材的契约检查；真实 RAW 测试另列 |
| fmt / cargo check / diff check | 通过 | 最终实现的格式、生产构建及 diff 检查 |
| 整库测试 | 963 passed / 0 failed / 40 ignored，176.10s | 最终构建实测；ignored 真实素材入口另行执行 |
| 9 NEF 原生 TIFF 单跑 | Rust 子进程及 runner 均 exit 0，native harness 通过 | 固定输入、构建、预算和原生路径；不是冷/热对照 |
| 导出契约与报告回填 | `pass` | `output.written=true`，实际路径、尺寸、位深、ICC、alpha 和 tag 338 均已记录 |
| 全 RGBA 样本 / ICC 对照 | 与 v17 精确一致 | 全量 827,128,680 B 未压缩 RGBA16 样本及 ICC 对照 |
| 几何 / 质量证据保持 | 10 项语义字段全部精确一致 | 仅排除 fusion 的 `graph_cut_seconds` 和 QG 的 `timing`，无浮点容差 |
| 独立解码器识别 | `sips` 和 `ffprobe` 均通过 | alpha、16-bit、sRGB、尺寸及 `rgba64le` 识别正确 |
| 综合导出审计 | 14 项全部为 true | [`sleeping-9-v18-export-acceptance.json`](benchmarks/sleeping-9-v18-export-acceptance.json) 中逐项证据 |

正式 TIFF 为 **15701×6585、16-bit×4 samples**，ICC 480 B，实际 `ExtraSamples=[2]`，文件大小 827,134,190 B，文件 SHA-256 为 `b420ad7c4926e025f7ede6846031997a8fa33c37fe2181af92a76b0f723313c0`。生产 IFD/header gate 本身不解码整幅图像；另外的有界脚本按逻辑行顺序比较未压缩 TIFF strip 样本，完整 RGB 与 alpha 的 827,128,680 B 均精确一致，像素 SHA-256 同为 `913e68fc8e1faef4b922f6b9e78753778c51bd5d896594e708f615d0180acda3`。它验证的是这对标准、未压缩 TIFF；BigTIFF 的正确声明与像素保持由定向测试覆盖。

ICC bytes 亦完全一致，SHA-256 为 `c56e1685d888f5edb92fe07f2750f387f8fe8e91b32ff8fb0b56bfbbb9458353`。容器的 strip 布局由 824 个 strip 变为 599 个，且新增 alpha 声明，因此文件 SHA 变化符合预期；不能把容器字节不同判为图像像素变化。`sips` 独立报告 `hasAlpha: yes`、16-bit、sRGB 和相同尺寸，`ffprobe` 报告 `rgba64le`；这两项格式解释与完整样本比较分别保留。

非 output 证据对照包括 grouping、intra_station、virtual_tiles、station_relations、closure、residual_warp、tone、composition，以及只排除计时的 fusion 和 Quality Gate。10 项均按排序键的精确 JSON 语义 SHA 核验，不使用近似浮点容差。新增 `output` 字段正确记录正式文件路径、尺寸、16-bit、sRGB、`alpha_preserved=true`、`tiff_extra_samples=[2]`、`artifact_validation=pass`，错误字段为空。完整证据见 [正式构建记录](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v18-export/verified/build.json)、[执行清单](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v18-export/verified/execution.json)、[全样本比较](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v18-export/verified/fullpixel-comparison.json)、[双解码器检查](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v18-export/verified/cross-decoder.json)及[正式 TIFF](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v18-export/verified/run/sleeping-9-6479-s1-v18-export-verified.tiff)。

| 性能观测 | v18 正式单次原生回归 |
| --- | --- |
| Rust 全程 wall time（含 TIFF/JPEG 导出） | 353.521472s |
| pipeline total / render | 348.986940s / 332.011383s |
| native fusion 次数 / 累计耗时 | 6 / 171.290324s |
| native fusion cache hit / miss | 15 / 6 |
| RAW decode | 70 次 |
| Stack_Report peak RSS（stitch/QG） | 11,325,423,616 B（10.5476 GiB） |
| OS 全进程 maximum RSS（含导出） | 11,333,681,152 B（10.5553 GiB） |

该表来自最终正式单跑，不混入报告漂移首跑。OS 峰值使用单个固定 Rust 子进程的 Darwin `getrusage(RUSAGE_CHILDREN)`，报告峰值仍只覆盖 stitch/QG。没有清理 OS 缓存，也没有进行新的受控冷/热对照，不能把与 v17 的耗时差异解释为本次导出修改带来的性能收益。RAW 解码仍为 70 次，来源缓存阶段尚未优化。

最终 `export_phase=pass`、`artifact_contract=pass`，但 `complete_native_quality_acceptance=false`。QG 仍为 pass、8 个 station owner evidence ROI；`DSC_6484.NEF` 仍由原生空间支持门禁排除，只保留 8 个有效来源。degradation 的 grouping motion、registration failure、成员排除和低清晰区记录均保持，MTF/boundary 仍为内容不适用。由于正式 TIFF 的全 RGBA 样本和 ICC 与 v17 精确相同，本阶段可复用原有 31 个窗口的视觉记录；这是有完整像素证据支持的复用，**没有宣称本轮重新逐窗人工检查**，也没有扩大此前检查覆盖范围或解决印章软边、外缘 coverage 缺口。

#### 21.5.1 首跑报告数字低位漂移：已修复，参考记录保留

首跑使用 SHA-256 为 `c9a6eb8b69e2fc2e1a00d872b6f7264535054170ed2b0e78eb1b3ca21ec8bedc` 的构建。其 TIFF alpha 声明和完整 RGBA 样本保持已经通过，但独立对照发现非 output 的几何、质量等报告证据出现 **571 处低位数字变化**，见 [`report-roundtrip-difference.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v18-export/report-roundtrip-difference.json)。这些变化来自导出回填时把整个 JSON 反序列化为带 `f64` 的类型再序列化，某些十进制 token 在往返后改变一个低位。图像样本一致不能替这类证据漂移背书，也不能把它归因于本轮没有修改的配准或融合算法。首跑继续标记为 [`report-roundtrip-defect-reference`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v18-export/first-run-classification.json)，`use_as_final_acceptance=false`；其耗时和内存不作为最终通过数据。

修复后的 `report_io::rewrite_stack_report` 使用 `serde_json` 的 `raw_value` feature，只解析和更新所需的 `output` 字段，非 output 顶层证据作为 `RawValue` 原样保留。未知顶层字段、未知 output 字段，以及 output 内未变化的值（包括 preview 证据）也保留原始 JSON 值片段；只有新增或实际修改的 output 字段重新序列化。外层字段顺序或排版允许变化，要求的是未修改证据片段的精确保留，不宣称更新后的整份报告字节与更新前相同。

新增 2 项定向测试验证了成功的 result-id、产物成功和产物失败回写保留原始非 output / preview 与未知字段，拒绝的 output 回写保持整份文件不变；最终 963 项整库测试和 `verified/` 中的真实 9 NEF 回归已包含并验证此修复。最终全样本、ICC 和 10 项几何/质量语义对照均通过，首跑发现的低位漂移不再出现。验收继续要求实际 IFD、像素、报告证据同时满足契约，不以近似浮点相等、QG 总体 pass 或文件存在替代。

### 21.6 后续配准与性能阶段：已定位，尚未修复

**第一步：补准原生配准的观测，再修复支持下降。** 同一份 `DSC_6484.NEF` 在 v16→v17 的粗/细 pass 接受数由 `3800/1838` 下降至 `381/117`，NCC 中位数由 `0.951/0.782` 下降至 `0.682/0.479`；控制格数接近，95px 间距仍在现有 112px 上限内。现有证据支持优先调查匹配内容和采样过程，不支持放宽 0.20 覆盖门槛。

只读检查发现确定的统计问题：`native_control_points.rs` 在调用反向 matcher 前设置 `reverse_attempted=true`；反向匹配失败返回时 `round_trip_px` 保持默认 0，但 `SimilarityStats::absorb` 会将其计入闭环距离分位数和 within-threshold 数量。实际 `accepted` 仍为空，所以当前门禁没有因此放行；但日志中的 0px 和“阈值内点数”不能全部解释为真实完成的闭环测量。后续先区分 reverse attempted/completed，并给前向、峰值、反向、闭环、邻域拒绝分别计数；以失败反向匹配的负例测试保证它不进入闭环分布。这项修复本轮尚未实施。

原生支持下降的待验证方向是采样前低通：`analysis_pair` 直接点采样到约 1400px 后才 blur，native `spacing=2` 也先隔点采样；scale=0.5 的源缩放已有面积/三角低通，因此两者的高频织纹和噪声条件不同。应以已知位移的织纹+失焦合成测试，以及冻结 station0 的真实 A/B 判断是否需要对称、mask-aware 的采样前低通。保留原生细化和现有门槛，不模糊最终输出，也不把该方向写成已经证实的唯一根因。

已有 `real_saved_capture_station_from_env` 可只渲染 station0（6483/6484/6485）。本次整管线运行已通过 `RAW_EDITOR_STACK_ALIGNMENT_CACHE` 保存 [`source-alignment.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v18-export/verified/source-alignment.json) 中的 source-stage homographies 和 focus warp，供该短入口复测；短入口的曝光及缓存 loader 与整管线并不完全相同，必须先证明它重现原始失败，才能用于评价修复。来源只有达到原门槛才可重新参与融合，报告继续保留明确排除信息。

**第二步：减少 owner 原图 payload 重建。** `load_tile` 在命中 NativeFusionCache 后，仍为每个实际 owner 重建 `FocusTileSources.originals`，逐源调用 `load_render_image`；因此 15 次融合缓存命中并不避免这些原图加载。2.4 GB 源缓存只能驻留约 4 张，跨 3 个 station 循环会触发淘汰和重解码。cache hit 还会将约 552 MB 的完整 RGB32F 图像 clone，需要把复制成本与 RAW decode 分开观测。

后续 trace 至少区分 matching-fusion、composition-fusion、composition-owner-payload 与 QG，并带 station、source、key、hit/miss、淘汰对象、解码/复制字节与时间。在既有字节预算内优先验证请求顺序、批次及受限共享 payload；不得跳过 originals，因为最终 owner 逆映射仍须从真实来源取样。本轮没有修改源缓存策略，也没有宣称 70 次解码已减少。

**第三步：补齐源 RAW 对照与最终冷/热验收。** 导出最终 station→output 映射，定位低清晰区与更多有效接缝，再对纸边印章等软边执行最佳焦点 RAW 同坐标对照。待配准及缓存修改稳定后，使用同一组 9 张 NEF、新固定构建和一致预算重新执行原生冷/热；分别记录产物契约、来源参与情况、QG/N/A、degradation、视觉覆盖与性能结果，不以单一 pass 字段替代完整结论。

## 22. v19：原生站内配准支撑修复与独立诊断（冷/热及视觉取样完成，整体验收未通过）

本阶段接续第 21.6 节，先处理 `DSC_6484.NEF` 的原生配准失败，再推进 owner 原图 payload 的用途分离。halo-only 修复的真实短入口仍未达到来源加入门槛；加入 spacing=2 采样前滤波后，已完成最终固定 release 的 977 项整库测试和同 9 NEF 原生冷/热。两跑中 6484 均为 `local`，density 保留 9 个有效源，现有 QG 和 TIFF 契约均通过，TIFF 字节一致。**整体原生质量验收仍为 false：degraded、MTF 样本不足、boundary N/A 和最终逐 RAW 像素归属限制保留；已完成的视觉取样确认印章软边及分离透明细孔，性能也没有优于 v18。** 下文区分短入口、正式冻结构建和后续尚未验收的优化，不改写 v17/v18 的历史结果。

### 22.1 已复现基线与对照边界

使用第 21.5 节的固定 v18 binary 和已保存的 `source-alignment.json`，单独执行 station0 短入口，已重现 `DSC_6484.NEF` 低于 0.20 空间支持门槛并被排除。station0 的缓存成员为 6484、6483、6485，group anchor 为 6483；旧短入口输出与执行记录保存在 [`station-baseline/`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v19-registration/station-baseline/)。

| 观测 | v18 station0 短入口基线 | v18 正式 whole 9 NEF 基线 |
| --- | --- | --- |
| `DSC_6484.NEF` 状态 | `failed` | `failed` |
| inliers / evaluated control points | 284 / 4845 | 412 / 4845 |
| inlier area coverage | 日志 0.059，三位小数 | 报告 0.0850361197110423 |
| median symmetric error | 日志 0.58px，两位小数 | 报告 0.5303300858899106px |
| control point spacing | 日志 95.0px，一位小数 | 报告 95.00571428571429px |
| 来源参与 | station0 融合 2/3 | output density 保留 8/9 个有效源 |
| shared display gain | 未应用 | 日志 `1.810`，仅显示三位小数 |
| Quality Gate / 原生导出验证 | 未运行 | QG `pass`、artifact contract `pass`，degradation 仍为 `degraded` |

两条入口共享基础 RAW loader，并从同 9 个源计算物理曝光 reference，但短入口的 `load_prepared_stack_source(...).image.to_rgb32f()` 没有执行整管线后续 shared display gain 和 clamp，也没有相同的 SourcePreflight/source cache/native fusion cache 生命周期。现有 alignment cache 不保存精确 display gain，不能用日志四舍五入后的 `1.810` 代替。因此这两列只共同确认旧失败可复现，不构成完全相同像素输入的数值对照。

首次 halo-only 短跑保持原 pixel loader，用旧短入口对候选短入口做同路径诊断；后续补充可选的精确 shared normalization，见第 22.4、22.7 节。完整 9 NEF 另行对照正式 v18。旧短入口日志未记录 anchor 的 registrar frame，摘要对该帧保留 `not_observed`，不从 group anchor 自动补写注册通过。旧基线包含 PNG、coverage 和 preview 导出；候选短跑可以跳过这些产物，其总 wall time 不能直接作为算法提速比例。

### 22.2 反向匹配 halo：half=33 调整为 40

已定位的确定缺陷是反向 patch 支撑不足。当前 NCC 匹配窗口半径为 25，即 51×51；搜索偏移范围仍为每轴 ±7 patch samples。旧 buffer half=33、side=67，只足够以 patch 中心为基点的正向搜索。反向 matcher 的基点是正向找到的 `found.target`，它已经相对原中心移动；反向仍需完整的窗口、搜索支撑和双线性插值邻点。旧 buffer 会使部分处于合法搜索范围内的位移在反向阶段尚未评分就因越界返回。

[`INTRA_STATION_PATCH_HALF`](/Users/dp/Desktop/raw-editor/src-tauri/src/panorama_utils/stack_pipeline/intra_station.rs:111) 现按 `MATCH_RADIUS + 2 × SEARCH_SAMPLES + 1` 计算，即 `25 + 14 + 1 = 40`，buffer 调整为 81×81。补充的 halo 提供反向所需采样，并未扩大正向或反向的 ±7 搜索偏移范围。粗 pass 的 `spacing=2` 下，既有 `(25+7)×2=64` 原生像素搜索支撑契约保持；完整采样 buffer 的范围与可搜索位移必须分开解释。

| 约束 | 本阶段状态 |
| --- | --- |
| 匹配窗口 / search samples | 保持 radius=25、51×51 / ±7 |
| NCC / second-peak margin | 保持 0.70 / 0.006 |
| 双向闭环 / 邻域中位位移容差 | 保持 1.00px / 4.0px |
| symmetric error / inlier area coverage | 保持 anchor 长边的 0.01 / 最低 0.20 |
| 控制点最大间距 | 保持 112px |
| legacy mosaic buffer | 保持 half=28、57×57；未扩大该路径 |
| owner/source、coverage、QG、TIFF 契约和 RSS 门禁 | 保持现有要求 |

该修改仍有需要真实验收的混合效应。`match_focal_plane` 在整个采样 patch 上估计高频能量并执行匹配低通，因此 67×67→81×81 会改变参与能量估计的内容和滤波边缘支撑；即使 NCC 核心窗口相同，后续样本、相关性和选择出的 sigma 也可能变化。更大的 buffer 还要求更大的有效来源覆盖，靠近 mask/画面边界的点可能转为 `patch_unavailable`。单个 patch 的采样面积从 4489 增至 6561，约增加 **46.2%**，可能增加采样和 focal-plane 处理成本；这不是整个进程 RSS 或总耗时必然增加 46.2%。

因此候选接受数变化不能全部归因于“反向越界消除”，也不能仅凭 buffer 修复排除采样前低通/织纹混叠等其他因素。验收同时读取 forward/reverse、patch unavailable、focal-plane 能量比与 sigma、邻域过滤和最终面积支持，保留这些效应的区分。最终图像不因本次诊断额外模糊。

### 22.3 消除失败反向匹配的假 0px 统计

旧 `reverse_attempted=true` 表示反向 matcher 已被调用，不代表它返回了有效匹配。失败分支保留默认 `round_trip_px=0`，此前会把这个默认值加入闭环分位数和阈值内计数。修复后只有 `measured.is_some()`，即双向匹配确实完成，才进入闭环距离和对应 peak-margin 分布；真正测得的 0px 仍正常计入，完成测量后被闭环/全局位移门槛拒绝的点也保留其真实距离。

新增观测区分 reverse attempted/completed/rejected，并按 correlation bucket 同时输出 closed/completed/attempted；另记录 patch unavailable、forward unmeasurable/correlation/peak/other 拒绝、roundtrip/global/neighbor 拒绝及邻域过滤前后接受数。统计修复不改变原有 `accepted` 门禁，也不将失败测量改为通过。新旧闭环分布的计数口径已经变化，不能将旧日志中过多的 0px 与修复后的分布直接解释为配准精度变差。

这些计数是**每个控制格最终采用的 measurement ledger**。高对比位置的测量失败后，代码可能回到格中心重测；`SimilarityStats::absorb` 汇总的是最终采用的记录，并未累计此前所有 probe 调用和重试。因此 attempted/rejected 不能充当底层 matcher 调用次数或重试开销的精确 profiler，性能结论仍需融合计时、分阶段 trace 与 RAW/复制计数支持。

### 22.4 短入口 JSON、归档和可比性声明

[`real_saved_capture_station_from_env`](/Users/dp/Desktop/raw-editor/src-tauri/src/panorama_stitching/acceptance_tests/focus_stack_fixture.rs:133) 已增强运行边界与证据输出。获取 run scope 后显式清空 degradation、intra-station、focus-fuser、residual、tone、topology 和测试 owner oracle，保持与生产入口一致；run scope 本身只负责串行访问，并不会清空旧 ledger。

- `RAW_EDITOR_NATIVE_STATION_SKIP_ARTIFACTS=1`：跳过 coverage PNG、完整图像、preview 和可选 ROI 输出及其额外来源解码；仍执行原站内融合并保存 JSON。
- `RAW_EDITOR_NATIVE_STATION_DIAGNOSTIC_JSON`：指定结构化证据路径；未指定时写到原 output 旁的 `*.diagnostic.json`。JSON 在大图导出前写入。
- `RAW_EDITOR_NATIVE_STATION_PRODUCTION_NORMALIZATION=1`：顺序准备同一 selection 的全部 9 源，通过生产 `raw_display_channel_percentile` 和 `shared_stack_raw_display_gain` 计算精确 f32 gain；每源准备一次，仅保留当前 station 的 3 个分析图，不保留全部原生 RAW payload。render 和可选 ROI 复用相同的 gain/clamp。默认仍只准备 station3、不施加 display gain；开启该模式时拒绝同时继承 fixed/skip/disable-shared normalization 开关。
- JSON 记录 registrar/fuser/degradation 快照、来源和 station 元数据、准备尺寸、owner legend/像素数、source refinement ids、准备/融合时间，以及明确的 `diagnostic_only=true`、QG `not_run` 和 source SHA 未验证标记。新模式另保存逐源 high、gain、各自 f32 bits、是否应用及 active prefilter policy；整管线像素等价仍标为 `not_verified`。

默认图像输出保留。短入口仍使用原通用图像保存分支，没有改为生产 TIFF writer；短入口 JSON 不是 `Stack_Report`，不能用测试入口返回 0、图像存在或 JSON 已生成代替完整原生验收。

统一提取入口为 [`summarize-registration.py`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v19-registration/summarize-registration.py)。`--base` 保留原 v19 短入口和实验目录，`--whole-base .../prefilter-release` 单独指定正式冷/热与该构建 manifest，避免混用 halo-only 和 prefilter 构建。正式运行脚本和 wrapper 位于 [`prefilter-release/`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v19-registration/prefilter-release/)，正式摘要为 [`prefilter-release/summary.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v19-registration/prefilter-release/summary.json)；原 [`archive/summary.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v19-registration/archive/summary.json) 保留早期阶段快照。提取器对尚未生成的候选保持 `pending`，不补零值或推导整体 pass。

完整阶段从各 `phase/run/` 的 Stack_Report、日志及 execution 清单提取 6484 和全部帧的状态/inliers/coverage/spacing/error、融合次数及耗时、RAW decode、source/native/ROI cache、QG ROI 与 station owner evidence、criterion N/A/技术不可测、closure/density、degradation 和 output contract。采样 RSS 与 `NATIVE_PROCESS_USAGE` 的 OS 子进程峰值分列。`Focus ownership solved` / `graph_cut_seconds` 使用 **StationFusion 内部累计时钟口径，不等于纯图割内核或完整 render/fusion wrapper 时间**：clock 从 `StationFusion::new` 开始，`fold_candidate` 记录其 elapsed，因此可包含两次 fold 之间的注册、取样等耗时。本次 A/B 始终比较 diagnostic 的 `timing.render_seconds`；preview/ROI 导出差异也不能隐藏在统一的 wall time 中。

### 22.5 当前验证进度与后续顺序

首轮 halo-only 固定二进制为 `/private/tmp/raw-editor-v19-registration-binary`，SHA-256 为 `fadc8adfcab947fa47569a7fc53f2c66827a267e9ebed19e69361dc578202a14`。[`build.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v19-registration/build.json) 保存该阶段 release 构建命令、源文件摘要和验证结果；它尚不包含后续 prefilter 实现，不能用其整库结果替后续构建背书。

| 验证项 | 当前状态 | 适用结论 |
| --- | --- | --- |
| 旧 station0 真实短入口基线 | 已复现 | 6484 failed，284/4845、日志 coverage=0.059 |
| reverse halo 定向测试 | 2 项通过 | 正负位移、spacing=1/2 和既有搜索边界；旧 buffer 反向支撑失败可重现，新 halo 可完成合法闭环 |
| 拒绝边界覆盖 | 包含于上述定向测试 | 超出 ±7 的位移、周期歧义纹理和不相关内容仍受原门禁拒绝 |
| halo-only union coverage 属性验证 | 100 cases 通过 | 保持该构建覆盖的 union 契约；后续 prefilter 的 fixture 调整另见第 22.7 节 |
| false 0px 统计修复及负例测试 | 6 项通过 | 反向失败不再计作已完成的零距离闭环，真实零距离及已完成但被拒绝的测量保持正确口径 |
| halo-only release 构建 / fmt / cargo check | 已完成 | 固定候选二进制及 SHA 已归档 |
| halo-only release intra-station / runner / 数据集契约 | 25 项 / 8 项通过；实际数据存在性检查通过 | 定向回归和输入契约通过；不能替代真实 RAW 处理 |
| halo-only release 整库测试 | 971 passed / 0 failed / 40 ignored，104.57s | 只对应上述冻结 SHA |
| halo-only station0 真实短跑 | 完成，6484 仍 failed | 687 inliers，coverage=0.14179566563467494，owner 像素仍为 0 |
| normalized prefilter station0 A/B | 完成，诊断范围内恢复参与 | 1028 inliers、coverage=0.21217750257997936，6484 拥有 13,840,540 像素；QG 未运行 |
| prefilter 定向回归 | 共 37 项通过 | 6 项新增采样测试 + 6 项统计测试 + 25 项 intra-station；另有 period-4 union property 100 cases 通过 |
| 最终 prefilter release / 整库测试 | 977 passed / 0 failed / 40 ignored，108.91s | 对应 e4dce3a0… 冻结构建，不沿用 halo-only 971 项结果 |
| 同 9 NEF 原生冷/热 | 已完成，QG / artifact contract 均 pass | 两跑 density 为 9 个有效源，TIFF 同 SHA；degraded 与 N/A 另列 |
| 原生视觉 / 最终每 RAW 像素归属 | 视觉取样完成但未通过整体验收；最终归属 pending | 已确认印章软边与分离 alpha=0 细孔；不用 station-local owner 数或 ROI descriptor 代替最终全图归属 |

**第一步补齐配准修复的来源证据，并定位视觉缺陷。** 正式冷/热已确认 6484 满足原 0.20 面积支持及其余注册门禁，closure、density、QG 和 TIFF 结果见第 22.8 节。视觉取样已确认分离透明孔，须定位其 coverage/owner 写入链，并对软印章做同坐标来源对照。仍须从最终 owner plane 读取每 RAW 有效覆盖像素，并将实际最终 QG 迭代的只读来源窗口与 TIFF 裁片绑定；station0 的 13,840,540 像素不能直接搬成最终全图贡献。

**第二步落实 payload 用途分离与 CPU 成本定位。** 将 matching-fusion、composition-fusion、composition-owner-payload 和 QG 的来源需求、解码/复制字节及耗时分别观测；再减少 native fusion 命中后的重复 originals 重建，验证受限共享、请求顺序及 prefilter 的 CPU 成本。保持真实 owner 逆映射及现有字节预算，不能省略必需来源取样。正式 v19 的 RAW decode 为 85 次，当前没有提速结论；用途分流、CPU 优化和只读 capture 属后续阶段，尚未由本次冻结冷/热验收。

**第三步对后续优化重新执行固定配置验收。** 继续使用同 9 张 NEF、`scale=1`、相同缓存预算和生产 TIFF writer，将性能运行与 capture/离线视觉运行分开。当前 v19 两次计时和选定原生窗口取样已完成，但 6484 最终贡献区域、所有接缝与全图覆盖尚未完成验证，透明孔与软边问题也未解决；候选像素已随配准改变，不能复用 v18 的旧视觉通过记录。只读 capture 的 reference 是已选 owner 对照，也不能称为全部候选中的最佳焦点 RAW 证明。

### 22.6 halo-only 真实短跑：支持提高，来源仍被排除

首轮固定 release 的 [`station/station.diagnostic.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v19-registration/station/station.diagnostic.json) 显示，6484 的 inliers 从旧短入口 284 增至 687，coverage 从日志 0.059 提高到精确值 `0.14179566563467494`，但仍低于 0.20，状态保持 `failed`。owner legend 虽然有 6484 的槽位，其 owned pixel count 为 **0**，有效 `source_refinement_ids` 仍只有 `[4,6]`；不能把路径出现在 legend 或被读取过当作来源恢复。

此处 coverage 分母是 anchor 的 4845 个有效控制格。日志中的 4844 是实际 evaluated control points，因一个格子的 patch 不可用而少一；它不是 coverage 分母，不能直接用 `687/4844` 重算并替换报告面积支持。原生间距仍为 `95.00571428571429px`，median symmetric error 为 `0.5762215285808054px`。

结果证明扩大反向支撑消除了部分失败，但不足以独立恢复第九帧。旧 short 与 whole v18 本来就存在 display gain 差异，因此下一轮先补齐精确 normalization，再在同一 binary 上开关采样前滤波；未把未经归一化的 release 短跑与后续 debug 短跑差值全部归因于 gain 或某一个算法变化。

### 22.7 spacing=2 采样前滤波：同构建、同 gain 的真实 A/B

新路径仅作用于 `IntraStation` 的 `spacing=2` 原生控制点采样。在每个原采样中心的原生坐标 `(x±0.5,y±0.5)` 取 4 个等权 tap，每个权重 0.25；在隔点采样前平均，保持常量响应和采样中心。base 与 source 使用同一组对称偏移，缺失支撑时整对测量不可用，不零填充，也不分别按剩余 tap 重归一化。

每个 tap 额外检查其实际需要的完整插值支撑：内部 cubic stencil、整数快速路径或完整边缘 bilinear stencil。mask/source footprint 任一必需样本缺失，或取样需要越界 clamp，均拒绝该对。因而本次 A/B 的完整变化是 **4-tap prefilter 加严格插值支撑检查**，不是只增加一个平均算子。`spacing=1` 和 legacy mosaic 保留原采样路径，全部注册/QG 门槛保持第 22.2 节所列值。

仅测试构建提供 `RAW_EDITOR_STACK_DISABLE_NATIVE_PREFILTER`，该变量存在即回到旧 point sampling；生产构建忽略此诊断开关。两跑使用相同 `/private/tmp/raw-editor-v19-prefilter-debug-binary`，SHA-256 为 `95c032d996231957bddcec650293e234bfbaafe9d42ec37f8368fa8dfaa12883`，profile 为 `test optimized + debuginfo`，不是 release 性能验收。

两跑均打开精确 production normalization，从全部 9 源计算 shared gain **1.8099923133850098**，f32 bits 为 **1072147924**。独立 [`experiments/ab-summary.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v19-registration/experiments/ab-summary.json) 核验了 binary SHA/运行后未改变、alignment SHA、物理曝光 f64 bits、逐源 high/f32 bits、gain/f32 bits、station inputs 和 global transforms 一致；两跑 exit code 均为 0。这个检查没有重新证明源 RAW SHA，也没有执行整管线像素等价比较，二者仍保留未验证状态。

| station0 / 6484 观测 | normalized halo-only | normalized prefilter |
| --- | --- | --- |
| 原生注册状态 | `failed` | `local` |
| inliers / inlier area coverage | 727 / 0.1500515995872033 | 1028 / 0.21217750257997936 |
| control point spacing / median symmetric error | 95.00571428571429px / 0.5762215285808054px | 相同 |
| coarse / fine pass 接受数 | 611 / 264 | 918 / 268 |
| coarse NCC p50 | 0.682 | 0.758 |
| coarse patch unavailable | 102 | 127 |
| 6484 owner 像素 | 0 | 13,840,540 |
| source refinement ids | `[4,6]` | `[4,5,6]` |
| 全 station covered pixels | 46,171,581 | 46,464,801 |
| preparation / render | 16.922991042s / 34.248177917s | 17.074197042s / 54.274495792s |
| Quality Gate / 性能验收 | `not_run` / false | `not_run` / false |

真实证据分别保存在 [`normalized-halo/station.diagnostic.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v19-registration/experiments/normalized-halo/station.diagnostic.json)、[`normalized-prefilter/station.diagnostic.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v19-registration/experiments/normalized-prefilter/station.diagnostic.json) 及同目录 execution/log。6484 通过的是原门槛，并实际贡献 station0 像素；这比仅观察状态变为 local 更完整，但仍不证明它已参与最终 whole 9 NEF 原生 TIFF。

coarse 接受数和 NCC 提高，fine 接受数仅小幅变化，细采样的 correlation 拒绝仍较多。严格支撑也使 coarse unavailable 从 102 增至 127，并未伪造边界有效样本。6485 仍为 local，但由于 6484 此时已参与前序合成，6485 看到的 base 也改变；其指标差值不能作为一个与其他帧无关的单帧滤波消融结果。

render 时间增加 **20.026317875s（58.47%）**。两跑虽同 debug binary 和 normalization，但 prefilter 路径增加采样工作，并多融合一张此前被拒绝的有效帧；不能把全部增加量解释为滤波单算子的成本，更不能宣称性能已提升。后续必须通过固定 release 的完整 9 NEF 测量 fusion、RAW decode、payload 复制、QG、导出和峰值内存。

新增 6 项 prefilter 测试覆盖常量/斜坡中心与双边对称、完整 cubic 支撑及禁止部分重归一化、fine/legacy 样本逐位保持、织纹降采样、带失焦的正负亚像素位移，以及周期/不相关内容拒绝。连同原 6 项统计测试、25 项 intra-station，共 **37 passed**。

原 union-coverage 合成素材利用奇偶采样相位差构造 coarse/fine 的互补接受；新增采样前平均后，这个素材不再稳定提供原先约定的 coarse-only 证据。测试已改用共享非周期低频结构加反相的 **4 原生像素周期织纹**，在 coarse 阶段可滤除歧义、fine 阶段仍保留竞争峰，继续测试真实测量的跨 pass union，而不注入接受点、不修改生产门槛。修改后 property **100 cases 通过**，失败种子保留在 [`intra_station_registration.txt`](/Users/dp/Desktop/raw-editor/src-tauri/proptest-regressions/panorama_utils/stack_pipeline/properties/intra_station_registration.txt)。冻结 debug build 的早期 manifest 中仍有当时的 fixture pending 记录，应与后续验证分开阅读，不回写成该冻结时点已经完成。

上述 A/B 的结论限于 station0 诊断；后续正式 prefilter release、整库、whole 9 NEF 冷/热和生产 TIFF 契约已另行完成，见第 22.8 节。原生视觉取样发现的缺陷和最终每 RAW 归属仍需处理，完整 Quality Gate、N/A 和 degradation 按正式报告分别记录，不能由局部结果自动替换。

### 22.8 正式 prefilter release：9 NEF 冷/热结果与剩余限制

正式构建为 `/private/tmp/raw-editor-v19-prefilter-release-binary`，SHA-256 **`e4dce3a0ef48313a7bf1bd4deb2301119e61cf91ca54c1fc492bb239f6681c9a`**。它完成 fmt/check、977 passed / 0 failed / 40 ignored 的整库回归（108.91s）、8 项 runner 测试及实际数据集契约；与前面的 halo-only release 和 debug A/B 分别归档。证据见 [`prefilter-release/build.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v19-registration/prefilter-release/build.json)。后续 payload/capture 代码不属于此冻结 SHA，不能计入这组验证。

两跑继续使用 `DSC_6479.NEF`–`DSC_6487.NEF`、`scale=1`、coverage alpha、2,400,000,000 B source cache、4 GiB native cache 和原 24 GiB stitch/QG RSS 门禁。各为独立进程，OS cache 未清理，且此前诊断已读取过源文件；冷/热表示顺序，不表示磁盘冷启动。两次执行清单和 Rust 子进程均 exit 0，binary SHA 前后不变，结束后独立重读 9 个源文件 SHA 均未改变。

#### 22.8.1 配准、质量和来源证据

冷/热均记录 **3 个 anchor + 6 个 local，0 个 failed**。6484 的结果与 normalized prefilter 短入口一致：1028 inliers、coverage `0.21217750257997936`、间距 `95.00571428571429px`、median symmetric error `0.5762215285808054px`。输出密度的有效 source count 从 v18 的 8 恢复到 9；原生注册失败和成员排除项在两份正式 degradation ledger 中均已消失。

但 **station 接受不等于最终全图每 RAW 像素已核实**。13,840,540 是此前独立 station0 诊断的 6484 owner 像素数。正式报告只给出可用的 station 融合总 owned count；`virtual_tiles.provenance[].owned_pixels` 对所有 RAW 均为 0，不能作为有效逐 RAW 归属计数，既不能据此宣称 6484 最终为零，也不能拿短入口计数补写。QG 的 `roi_photometry.owner_path` 是 ROI 起点/descriptor，对 station-level 多 owner 块也不能代表整块。最终 owner plane 的每源有效 coverage 计数仍标记 pending。

| 正式质量观测 | 冷 / 热共同结果 |
| --- | --- |
| QG verdict / dense ROI | `pass` / 8 |
| station owner evidence / raw owner evidence | 8 / 0 |
| local scale median / pixel ratio | 1.0450645225–1.0957031250 / 8 个 ROI 均为 1.0，非 diagnostic |
| closure / direct-pair P95 | `converged` / 1.3685809081px |
| density factor / density 后 direct-pair P95 | 1.095703125 / 1.4995583779px |
| MTF | 512 候选、1 eligible/measurable、511 内容拒绝；仍 `not_applicable` |
| boundary | 372 候选、0 eligible；全为 low contrast，`not_applicable` |
| noise | 45 候选、24 measurable、21 内容拒绝；通过 |
| technical unmeasurable | 各 criterion 均为 0 |
| tone / degradation | `applied` / `degraded` |

MTF 唯一样本的归一化值为 `1.0000021694779606`，但样本数不足既有可测要求，不能改写为 MTF 通过。boundary 没有适用样本，也没有完成接缝笔画的这一项数值验证。degradation 仍含 `grouping_accumulated_motion` 4 条和 `fusion_low_sharpness_region` 6 条；QG 总体 pass 不清除这些记录。density 仍声明 `sampled_native_jacobian_not_full_pixel_proof`，不将采样局部尺度证明扩展为全像素证明。

#### 22.8.2 原生 TIFF、性能与冷/热一致性

| 指标 | v19 原生冷跑 | v19 原生热跑 |
| --- | --- | --- |
| TIFF 尺寸 / bit depth / samples / ICC | 15897×6720 / 16-bit×4 / sRGB | 相同 |
| output written / artifact contract / ExtraSamples | true / `pass` / `[2]` | 相同 |
| fusion 次数 / wrapper 累计耗时 | 6 / 236.433602s | 6 / 234.434630s |
| pipeline total / render | 418.561728s / 401.919904s | 409.841384s / 394.968736s |
| Rust 全程 wall（含导出） | 421.851967s | 412.561307s |
| RAW decode | 85 | 85 |
| source cache hit / miss / evictions | 68 / 76 / 72 | 相同 |
| source render hit–miss / QG hit–miss | 13–59 / 55–17 | 相同 |
| source cache peak / entries | 2,211,124,992 B / 4 | 相同 |
| native cache hit / miss / invalidations | 15 / 6 / 3 | 相同 |
| native cache peak bytes | 2,948,607,844 B | 相同 |
| QG reference 数 / ROI cache hit–miss | 33 / 0–33 | 相同 |
| QG source-loader decode 计数 / reference resampling | 13 / 20.483739s | 13 / 19.518619s |
| QG decode / ROI sampling 时间 | 7.911542s / 20.759661s | 7.769615s / 19.773326s |
| Stack_Report sampled peak RSS | 12,900,909,056 B（12.0149 GiB） | 12,401,688,576 B（11.5500 GiB） |
| OS Rust maximum RSS（含导出） | 12,920,078,336 B（12.0328 GiB） | 12,425,183,232 B（11.5719 GiB） |

SourcePreflight 的 stat/hash/EXIF 均仍为 9 次。85 次 RAW decode 对应 9 次准备加 76 次 source cache miss；QG `owner_source_decode_count=13` 是其 loader 调用口径，不能与缓存内部 QG miss=17 互相替代。native cache 仍有 3 次预期输入状态失效，matching/composition 各执行 3 次，另 15 次 composition 命中；无融合失败、淘汰或 bypass。

两份 TIFF 均为 **854,628,326 B**，SHA-256 同为 **`e16fd14781d86258cdeb7912abc43a3b230fa9935ec1488e0c1acda68901b605`**。独立 [`cold-hot-verification.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v19-registration/prefilter-release/cold-hot-verification.json) 核验 intra_station、closure、composition、virtual_tiles，以及 QG criteria、ROI、photometry、reverse-lookup、confidence 和 unmeasurable 等记录冷/热一致。文件一致支持这对运行结果可复现；没有替代人工视觉检查，也未声称其像素与 v18 相同。

相对第 21.5 节 v18 单跑，v19 冷跑 fusion 时间增加约 **38.03%**、pipeline total 增加 **19.94%**、Rust wall 增加 **19.33%**、OS RSS 增加 **14.00%**；RAW decode 从 70 增至 85。热跑相对 v18 也没有性能改善。此次保留 9 个有效源而非 8 个，密度和最终画布也改变，不能把所有成本增量归因于 prefilter 单算子。热跑 wall 较本次冷跑缩短约 2.20%，但 fusion 次数、解码数和缓存计数完全相同，不是跨进程内存复用或算法提速的证据。

#### 22.8.3 可追踪归档和未完成项

精简验收记录见 [`sleeping-9-v19-registration-acceptance.json`](benchmarks/sleeping-9-v19-registration-acceptance.json)，链接正式构建、完整报告、执行清单、timing、源身份与 TIFF SHA 验证；逐项保留注册、QG、N/A、degradation 和 artifact contract，不设置“全量 gate=true”。完整原生质量验收仍为 `false`。

原生视觉目录为 [`prefilter-release/visual/`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v19-registration/prefilter-release/visual/)。[`manifest.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v19-registration/prefilter-release/visual/manifest.json) 将 30 个原生 16-bit PNG 窗口绑定到上述 TIFF SHA；总览缩小到 2400×1015，只用于布局检查。根代理的 [`root-review.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v19-registration/prefilter-release/visual/root-review.json) 已检查 18 个窗口，另有 [`independent-review.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v19-registration/prefilter-release/visual/independent-review.json) 检查 15 个窗口，包含六固定区域、五个 QG context、三个报告 boundary context 和上部 coverage 台阶。按已保存窗口名核对集合，重复项为 `qg-roi-03`、`qg-roi-04`、`upper-coverage-step`，去重后 **30/30 个导出原生样本全部已查看**，无未阅导出窗口。新补阅的 `qg-roi-01`、`qg-roi-05` 仍可见软印章边缘，有限窗口内未见明显新增双轮廓；扩大 context 中的印章不能替代中心 ROI 的实际内容。结论是**全部导出样本取样复核完成，完整视觉验收未通过**：30 个窗口仍不是全图所有像素或全部来源候选。v19 原始记录中的最终每 RAW 归属仍缺失，v20 同 TIFF 的后续 census 另见第 23.8 节，不追溯改写旧报告。

已阅床架接头、中央床栏、凳脚与两鞋轮廓连贯，有限窗口内未见可确认的大尺度错断或平移双影。但左大印、左小印和右侧印章笔画在原生尺度明显偏软，织纹清晰度与横向条带也随区域变化。缺少同坐标来源对照，目前不能把软边直接归因于算法失焦，更不能宣称已保留最佳焦点。三个报告 boundary context 仍对应低对比 `content_not_applicable`；扩大 context 中出现印章不使原测量样本自动变为可测。

除了上、下、右侧的外覆盖台阶，独立复核解码了三个已有 RGBA16 PNG 的原始 alpha，确认以下**与外轮廓分离的透明细段**，其核对邻域 alpha 为 65535。坐标是最终 TIFF 的 0 起始 canvas 坐标，x 为闭区间；本轮只证明存在和位置，未证明原因或完成全图孔洞普查。

| 区域 | canvas y | canvas x（含两端） | alpha=0 像素数 |
| --- | --- | --- | --- |
| 上部细孔 1 | 522 | 9345–9348 | 4 |
| 上部细孔 2 | 523 | 9389–9398 | 10 |
| 上部细孔 3 | 524 | 9434–9447 | 14 |
| 上部细孔 4 | 525 | 9478–9497 | 20 |
| 上部细孔 5 | 526 | 9523–9533 | 11 |
| 右下邻近边界细孔 | 6459 | 15645–15648 | 4 |

上部合计 59 像素，在 `qg-roi-08.png` 和 `upper-coverage-step.png` 中是同一组孔，不能重复计数；加右下 4 像素，本次核对的唯一像素合计 **63**。`qg-roi-08` 的孔位于扩大 context 的 local y=138–142，不在从 local y=256 开始的中心 256px QG ROI 内。因此既不能将这些孔一概当作合法外覆盖轮廓忽略，也不能把 context 的缺陷误写为该中心 ROI 已测失败。

后续只读 capture 默认关闭，独立运行须绑定最后 completed QG/repair 迭代与 TIFF，并明确 reference 为选中 owner。v19 冻结产物未包含有效最终 owner census；该项与选中来源窗口绑定已由第 23 节的 v20 独立 capture 补齐，缺口链路定位和更广泛焦点候选验证仍待完成。用途分流、CPU 优化与 capture 不能追溯计入 e4dce3a0… 的 v19 冻结结果。

## 23. v20：按用途加载 owner payload 与等价采样优化（冷/热、capture 绑定与来源取样完成，整体质量仍未通过）

本阶段在 v19 的配准支撑和四 tap prefilter 基线上减少无用来源准备与重复支撑检查。冻结 release 为 `/private/tmp/raw-editor-v20-purpose-release-binary`，SHA-256 **`08f33e2d9c964fd5a09f1b67e36d2c571d9f73693f7eb8214ded52bbada010a6`**。已有 fmt/check、995 passed / 0 failed / 40 ignored 的整库回归（159.59s）、8 项 runner 测试和本地真实数据集契约；证据为 [`v20-purpose/build.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v20-purpose/build.json)。**同 9 NEF 冷/热已完成：与各自 v19 基线的正式等价检查通过，RAW decode 均从 85 降至 67；本次冷/热 Rust wall 分别减少 12.43% / 10.38%。** 第三次独立 source capture 也已成功完成，8 个原生窗口逐像素绑定最终 TIFF，并确认 6484 在最终覆盖中贡献 9,462,041 像素；包括 capture 在内的五份 TIFF 同 SHA。8 对选中 owner 来源窗口已完成视觉取样，仍不能证明最佳焦点；v19 的透明细孔、软边、degraded 和 N/A 未被本次等价优化修复。

### 23.1 分开 raster 请求与 owner source 请求

`FocusTileLoadPurpose` 明确区分 `RasterOnly` 和 `OwnerSampling`。曝光/规划等仅消费融合 raster 的调用仍获取完整融合结果及缓存中的 owner/source/refinement 证据，但不再额外扫描 ownership、解析 refinement 并重建暂时不会被读取的 originals payload。`tile_payload::load_owner_sources` 在 `RasterOnly` 分支直接返回；真正执行 owner 逆映射和最终来源取样的路径继续使用 `OwnerSampling`，不省略必需 RAW。

非零 owner 的 source/refinement 缺失和 loader 错误继续向上传播。零像素 legend 槽位仍保留为 `None`，不能压缩槽位而改变 owner 编号。该修改不增加跨运行共享，也未扩大缓存预算或删除融合结果中的 provenance。新增 `performance.tile_payload` 分列 `raster_only_requests`、`owner_sampling_requests`、`owner_payload_build_requests`、`owner_source_load_requests` 与 `raster_only_payload_skips`；请求在后续失败前也会计数，owner source load 可能命中缓存，**不等于 RAW decode 次数**。旧报告缺少本字段时按默认值反序列化，真实比较仍须区分旧版未观测与新版测得零值。

5 项 payload 定向测试覆盖 raster 分支零额外来源加载、零归属槽位、缺失 sampling/loader 错误、compositor 请求用途及原生 payload 像素保持，以及报告兼容。正式冷/热均观测到 6 次 raster 请求及 6 次 payload skip、12 次 owner-sampling/payload build、36 次 owner source load request；RAW 解码和成本变化见第 23.6 节。v19 没有本组 payload counter，不能把缺字段当零值计算增量。

### 23.2 保持 prefilter 数值与拒绝行为的 CPU 优化

spacing=2 的四个 `(±0.5,±0.5)` tap、0.25 权重、tap 累加顺序、严格 cubic/integer/edge-bilinear 支撑、halo=40 和全部注册门槛保持。每个 tap 的完整 source 逆映射只计算一次，支撑检查返回已算出的 source point，像素取样复用该点。四个 tap 间只跳过**已经完整验证通过**的支撑矩形中重复像素；base/source 各自持有最多四个局部矩形，生命周期仅为一次 area sample，不建立整图缓存。

去重不使用覆盖多个支撑的 bounding box 代替真实支撑并集，因此不会把间隔处的 mask 缺口、无效样本或黑边当作合法采样，也不扩大有效域。spacing=1 和 legacy 保持原路径。独立 scalar oracle 保留原重复映射和支撑检查，定向比较可用/不可用判定、f64 样本 bits、非平凡坐标/金字塔/边界条件，以及控制点测量、拒绝和 focal-plane 统计；`native_control_points` 18 项、intra-station 25 项已通过。同 9 NEF 的正式 registrar、几何、owner/source 报告证据、QG 和 TIFF 等价检查也已通过，详见第 23.5 节；两项优化在同一候选内验收，不能把总耗时差全部归给 prefilter 单项。

### 23.3 独立只读 source capture 与 TIFF 绑定

capture 仅在测试构建中且显式设置 `RAW_EDITOR_QG_SOURCE_VISUAL_DIR` 时启用，性能冷/热默认关闭。它复用当次现成 output/reference ROI，包括 reference cache hit；不增加 `load_source`、RAW 解码或重新采样，不修改 output、ownership、coverage、thresholds。默认最多 4 个窗口、可设最多 8 个，先每个 station 选一窗再填剩余；单边硬上限 512。此次独立 capture 配置最多 8 窗，覆盖结果仍以实际 manifest 为准，不能预先声称所有 station 已有证据。

产物包括未 resize 的 output/reference PNG16、精确 RGB f32LE、逐像素 owner PNG16、validity PNG8、prepared-source XY f64LE 和 ROI/world-origin/iteration/profile 元数据。station-level 多 owner 块按实际 `input.ownership` 对每像素走现有完整逆链，不以 descriptor 的中心 owner 代替整块；全图每源像素数只计 `coverage!=0`，明确保留零贡献源。reference 是现有 post-tone 的**选中 owner 对照**，不是所有焦点候选中的最佳焦点证明；prepared/developed 坐标也不等于 sensor CFA 坐标。

每轮开始先使旧 `latest` 失效并写 `running`，再创建本轮产物；启动失败写 `failed`、禁止离线绑定，状态 I/O 失败则尝试移除旧标记并明确报错，不能回退旧 complete 冒充最终结果。完成 QG 仅记录 completed iteration，不自行宣布整次导出成功。7 项 capture 定向测试覆盖多 owner、原生逆链坐标、无额外加载/缓存命中、关闭无副作用、跨 station 选择、迭代替换和 metadata 写失败。

[`bind-source-visual.py`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v20-purpose/bind-source-visual.py) 只在完整运行成功、latest complete 且无更晚尝试/失败时，将最后 capture 迭代绑定到最终 TIFF，另写独立 manifest，不改 raw capture。生产绑定显式要求 release、`debug_assertions=false` 和本次完整 8 个 ROI；缺窗不能借小合成测试配置放宽接受。capture/process/iteration/ROI 与每个 evidence 文件均检查解析后的目录归属并拒绝 symlink 串阶段；TIFF 的 inode/size/mtime/ctime 快照覆盖哈希前、哈希后与最终检查。

该脚本验证 PNG/f32/owner/XY 的尺寸、字节数、validity 和 source bounds，并从 TIFF strip 直接读取对应原生行，逐像素比较 RGB 与 alpha。RGB32F 已为 display-referred，生产转换为 f32 clamp、乘 65535、f32 round 后 u16；离线不再次施加 sRGB 曲线或 tone。alpha 为 coverage=0 时 0，否则 65535，并核对 production sRGB ICC 的字节 SHA。缺证为 `pending`，像素差异为 `mismatch`；`bound` 仅证明这些窗口的导出身份，不代表最佳焦点或完整视觉通过。小合成端到端与 debug/少窗/symlink/哈希区间变更负例已通过，正式 capture→TIFF 绑定也已为 `bound`，详见第 23.7–23.9 节。

### 23.4 同配置冷/热接受条件与当前状态

[`run-acceptance.py`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v20-purpose/run-acceptance.py) 串行执行 cold、hot、capture，全部使用同一冻结 binary、6479–6487 九源、`scale=1`、production TIFF/coverage alpha、2,400,000,000 B source cache、4 GiB native cache和既有 24 GiB RSS 门禁。每轮为新进程与新产物目录；OS cache 未清理，冷/热只表示顺序。capture 有额外 PNG/f32/XY/JSON I/O 及全图 owner 计数，**其 QG/total/wall 时间不得混入性能冷/热对比**。

| 验收维度 | 接受约束 | 当前状态 |
| --- | --- | --- |
| 固定构建与输入 | SHA 前后不变；九源身份与配置一致 | 三次 v20 运行通过；结束后另行重读九源哈希，Rust source manifest/binary 均与冻结记录一致 |
| 配准与来源 | 仍为 3 anchor + 6 local；6484 原门槛通过；density source count=9，owner/source 证据保留 | 冷/热与 capture 通过；capture 最终 owner census 九源均有正贡献 |
| 几何与质量 | 对比 closure、density、全部现有 criterion 记录、ROI、station evidence、N/A、degradation；不放宽 gate 或把 N/A 改为通过 | 对照通过；QG pass、degraded、MTF/boundary N/A 均保留 |
| 原生产物 | 16-bit RGBA/sRGB/ExtraSamples `[2]`；冷/热一致，并以与 v19 TIFF 的字节/像素一致为本次等价优化目标 | 两版冷/热与独立 capture 共五份 TIFF SHA 完全一致 |
| 性能观测 | fusion 次数/时间、RAW decode、source/native/ROI cache、tile payload 请求、pipeline/render/wall、sampled 与 OS RSS 分列 | 已完成，见第 23.6 节；仅本次单冷/热对照的观测 |
| capture 与来源视觉 | 独立运行成功、最终迭代绑定、逐 RAW 最终覆盖计数、多 owner 坐标和选中来源窗口核验 | 完成：iteration 1，8 窗绑定及8对来源取样；不计入冷/热性能，不代表最佳焦点 |
| 完整原生视觉 | 上部/右下细孔需定位复验；软印章、条带和焦点候选对照仍需证据 | 未通过，不随等价/性能结果自动变为通过 |

### 23.5 正式冷/热等价结论与保持不变的质量限制

独立 [`comparison-cold-hot.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v20-purpose/comparison-cold-hot.json) 分别完成 `v19_cold→v20_cold`、`v19_hot→v20_hot`、`v20_cold→v20_hot` 三组比较，`performance_equivalence_status=pass`、errors 为空。比较不仅覆盖 registrar/closure/composition，还包括完整 `virtual_tiles`、`grouping`、`topology`、`input`、`virtual_tile_cache`，以及去除 timing 的完整 QG、owner evidence、output contract、tone、degradation 与 memory budget。每组已比较 JSON 子树的哈希一致、changed leaf 为 0；运行目录/结果 ID 等身份字段与性能计时不伪装成相同业务值。

四份 TIFF 均为 **15897×6720、854,628,326 B**，SHA-256 同为 **`e16fd14781d86258cdeb7912abc43a3b230fa9935ec1488e0c1acda68901b605`**，即与第 22 节的 v19 原生产物字节一致。v20 哈希由本轮显式读取 TIFF 计算，并存入 stat-bound sidecar；v19 沿用其正式哈希验证记录，具体来源范围保存在比较文件中。

6484 仍为 `local`、1028 inliers、coverage `0.21217750257997936`，总计 3 anchor + 6 local；closure direct-pair P95=`1.3685809081403442px`，density factor=`1.095703125`、source count=9。QG 仍为 pass、8 个 dense ROI、8 个 station owner evidence、0 个 raw-owner ROI；MTF 仍只有 512 个候选中的 1 个可测样本，boundary 仍为 372 个候选中的 0 个可测，均保持 N/A。技术不可测仍为 0，degradation 仍有 4 条 grouping motion 和 6 条 low-sharpness 记录。原门禁没有变化；正式报告中旧 per-RAW provenance 计数的不足未被改写，补充的最终 owner census 来自第 23.8 节的只读 capture。

同一 TIFF 也意味着第 22.8.3 节已确认的 63 个透明细孔像素与软印章问题仍在。这里通过的是**本次性能修改的等价检查**，完整原生视觉/最佳焦点验收仍未通过，不能写为全量质量门禁全部通过。

### 23.6 正式冷/热性能：RAW 解码减少 18 次，融合次数保持 6

| 观测 | v19 冷 | v20 冷 | v19 热 | v20 热 |
| --- | --- | --- | --- | --- |
| RAW decode | 85 | 67 | 85 | 67 |
| fusion 次数 / wrapper 累计秒 | 6 / 236.433602 | 6 / 213.370829 | 6 / 234.434630 | 6 / 212.483939 |
| pipeline total 秒 | 418.561728 | 366.485711 | 409.841384 | 366.934562 |
| render 秒 | 401.919904 | 351.760724 | 394.968736 | 349.263916 |
| Rust 子进程 wall 秒（含导出） | 421.851967 | 369.421936 | 412.561307 | 369.730270 |
| sampled Stack_Report peak RSS（B） | 12,900,909,056 | 12,121,784,320 | 12,401,688,576 | 11,676,205,056 |
| OS Rust maximum RSS（B，含导出） | 12,920,078,336 | 12,132,499,456 | 12,425,183,232 | 11,792,138,240 |

相对同标签 v19，本次 RAW 解码均少 **18 次（21.18%）**；fusion wrapper 时间减少 **9.75% / 9.36%**，pipeline total 减少 **12.44% / 10.47%**，Rust wall 减少 **52.430031s / 42.831038s（12.43% / 10.38%）**，OS 峰值 RSS 减少 **787,578,880 B / 633,044,992 B（6.10% / 5.09%）**。fusion 仍为 matching 3 次 + composition 3 次，native cache 仍有 15 次命中、6 次 miss、3 次预期失效，并未通过少融合一站或排除来源换取这些减少。

| 计数 / 有界缓存观测 | v19 冷/热共同值 | v20 冷/热共同值 |
| --- | --- | --- |
| SourcePreflight stat / hash / EXIF | 9 / 9 / 9 | 9 / 9 / 9 |
| source cache hit / miss / eviction | 68 / 76 / 72 | 68 / 58 / 54 |
| source render hit–miss / QG hit–miss | 13–59 / 55–17 | 13–41 / 55–17 |
| source cache peak / resident entries | 2,211,124,992 B / 4 | 相同 |
| native cache peak / peak entries | 2,948,607,844 B / 3 | 相同 |
| QG reference / ROI cache hit–miss | 33 / 0–33 | 相同 |
| QG source-loader decode 计数 | 13 | 13 |
| raster 请求 / payload skip | 旧版未观测 | 6 / 6 |
| owner sampling / payload build / owner source load requests | 旧版未观测 | 12 / 12 / 36 |

67 次解码对应 9 次准备加 58 次 source cache miss；减少量集中在 render miss 的 59→41，QG miss=17 与其 loader 计数=13 均未减少。payload 用途分离的 6 次跳过与这条成本变化一致，但旧报告没有 payload counter，不能把缺字段补成 0，也不能仅凭这对运行将全部时间收益归因到某一个子改动。`graph_cut_seconds` 仍按第 22.4 节的 StationFusion 累计 elapsed 解释，不当作纯图割内核时间。

本结果是每版本各一次 cold/hot 的有限对照，未清理 OS page cache，也没有重复随机化测量或统计置信区间。v20 热跑 Rust wall 比冷跑高约 0.31s（0.08%），因此不把“热”标签当作必然更快或跨进程内存缓存复用的证明。上述 wall、RSS 改善仅陈述本次观测，不能据此给出所有素材的普遍提速承诺。

### 23.7 归档与正式 capture→TIFF 绑定

精简机器可读验收为 [`sleeping-9-v20-purpose-acceptance.json`](benchmarks/sleeping-9-v20-purpose-acceptance.json)，保留两版各自 cold/hot 数字、14 组等价子项及哈希、构建和完整报告路径、TIFF sidecar 证据，以及原有 N/A/degraded/视觉限制。它不把 capture 的额外计数与 I/O 耗时混入性能对照，也不将统计请求数改称解码数。

第三次 source capture 使用同一固定二进制独立运行成功，仍为 RAW decode=67、fusion=6，TIFF SHA 仍为 `e16fd147…b605`。最终 [`comparison-final-bound.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v20-purpose/comparison-final-bound.json) 的 SHA-256 为 `fe59aebb43c097e68342bf3ea3a3056b78ff1e15b19cf72126e90cd7a4bedb72`，其中性能等价为 pass、capture 为 pass、offline binding 为 bound、errors 为空；capture→hot 的 14 项业务/几何/QG/owner/output 等价检查全部通过。capture 的计时包含额外文件与坐标工作，即使本次 wall 数字较低，也不将其纳入第 23.6 节的两版冷/热性能收益。

独立 [`source-visual-binding/manifest.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v20-purpose/capture/source-visual-binding/manifest.json) 为 `status=bound`：最终采用 `process-18928/iteration-0001`，release、debug assertions=false、完整 8 窗，三个 station 均覆盖。每窗 output f32→capture PNG16、reference f32→capture PNG16、output f32→最终 TIFF RGB、coverage→TIFF alpha 均为逐像素 exact，未修改 raw capture。比较器现可信消费这份已保存的 binder proof，核对当前小证据文件哈希、身份/迭代及 TIFF stat-bound SHA 后输出 `offline_binding_status=bound`、`final_iteration=1`；本次归档提取未重新哈希大 TIFF，也未重跑像素绑定。原始 `latest.final_iteration=null` 按设计保留且不可被提取器改写。归档工具的 10 项测试已通过；上述原生逐像素结论仍来自此前实际 binder 执行，而不是这次只读归档提取。

8 窗均为 256×256，每窗 65,536 对 prepared-source XY 全部 validity=1，且每窗实际包含 2–4 个 owner。记录总量为 `8×65,536=524,288` 次 ROI 像素记录；**ROI00 与 ROI01 有重叠，不能称为 524,288 个独立唯一输出像素**。这证明既有选中 owner 逆链的有限窗口对应与范围检查，不能扩展为全图每像素几何或最佳焦点证明。

运行结束后的 [`final-input-source-verification.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v20-purpose/final-input-source-verification.json) 另行重读并确认全部 9 个 NEF 与 v19 身份相同，Rust source manifest 和固定 binary SHA 均未改变。这份结束核验与比较器仅使用报告中保存的源哈希是不同的证据口径。

### 23.8 最终 owner census：6484 确认贡献 9,462,041 像素

在已绑定最终 QG input owner plane 上，仅统计 `coverage!=0` 的 **97,858,218** 个像素；九个 RAW 均有正贡献，covered owner-zero 为 **0**，另有 8,969,622 个未覆盖像素。以下是最终原生 canvas 的来源归属，不再使用 station0 短入口的 13,840,540 像素代替最终贡献。

| RAW | 最终有效覆盖归属像素 |
| --- | --- |
| DSC_6479.NEF | 17,787,899 |
| DSC_6480.NEF | 11,893,911 |
| DSC_6481.NEF | 10,601,413 |
| DSC_6482.NEF | 11,369,535 |
| DSC_6483.NEF | 21,070,008 |
| DSC_6484.NEF | **9,462,041** |
| DSC_6485.NEF | 4,119,355 |
| DSC_6486.NEF | 8,983,597 |
| DSC_6487.NEF | 2,570,459 |

该 census 来自 v20 独立诊断，不追溯伪造 v19 报告中缺失的原始 owner plane。v19/v20 TIFF 相同提供结果身份联系，但来源归属记录的实际生成阶段仍明确为本次 capture。

### 23.9 选中来源取样复核与尚未解决的问题

[`root-source-review.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v20-purpose/capture/source-visual-binding/root-source-review.json) 已实际查看 ROI00–03；[`independent-source-review.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v20-purpose/capture/source-visual-binding/independent-source-review.json) 已实际查看 ROI04–07。8 对 output/reference 均为未 resize 的原生 256px 窗口。左侧印章软边同时存在于 output 和选中 owner reference；有限窗口中未见可确认的新增双影、断边或硬拼缝。这只支持选中来源对应，不能证明其他焦点候选同样软或 owner 选择最优。

[`output-reference-differences.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v20-purpose/capture/source-visual-binding/output-reference-differences.json) 独立记录量化前 display-referred RGB32F 差分，不新增质量阈值。ROI05/06 的 output 与 reference 原始浮点数据相同，其余六窗有差异；最大单通道绝对差出现在 ROI07，为 `0.0030503571033477783`，该窗 mean absolute difference 为 `0.0001795591452567654`。**这些是 output 与来源 reference 的差异，不是 capture output 与最终 TIFF 的绑定误差**；后者八窗均 exact。屏幕观察不能证明 16-bit 数值完全一致，也不能用小差分替代最佳焦点验收。

此前确认的 63 个透明孔像素均不在这 8 个精确 dense QG ROI 内，source capture 未修复也未豁免它们。下一阶段按其 canvas 坐标追查 coverage/owner 写入链并建立定向回归，再复验透明孔；软印章与横向条带则继续补实际相关焦点候选的同坐标比较。完整原生视觉/质量验收仍为 **false**，不因性能等价、已完成的导出窗口取样或选中来源绑定完成而改为总体完成。

## 24. v21：透明细孔的有界定点诊断（真实证据已绑定，63 点失败链已定位，修复待验收）

本阶段先回答第 22.8.3 节已确认的 63 个透明孔为何没有写入有效 coverage，再决定修复位置。真实 9 NEF 运行、8 窗 source capture 和 89 点 trace→TIFF 绑定已完成；独立重算定位到这 63 点共同的直接失败链：最近局部 owner 对应来源在当前连续目标坐标下不满足 footprint，native sampler 返回 None，同一 3×3 邻域内的有效替代来源没有被尝试。**冻结诊断版本仍保留全部 63 个孔，修复尚未验收。** 第 23 节已完成的 v20 性能结论保持原有范围，本次 trace/capture 运行不追加为性能冷/热样本。

冻结 release 测试二进制为 `/private/tmp/raw-editor-v21-trace-release-binary`，SHA-256 **`83a74b2eef821d4f326641bf6ea430b74bdc70064604e937a9f83fe5a1df1aa0`**。[`v21-holes/build.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v21-holes/build.json) 保存构建和源码清单：fmt/check 通过、4 项定点 trace 测试通过、全库 **999 passed / 0 failed / 40 ignored，157.22s**。这些结果证明当前诊断实现通过回归，不证明真实孔洞原因已确定或已经修复。

### 24.1 输入点的来源：63 个孔、24 个不透明对照、2 个外覆盖负对照

[`prepare-points.py`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v21-holes/prepare-points.py) 只读取 v19 已导出的原生 RGBA16 PNG，不加载 RAW、不重新解码大 TIFF。它核对 visual manifest 中的 PNG SHA，再直接检查原始 16-bit alpha，未用 RGB 黑色或屏幕外观推断角色。复用 v20 binder 的 PNG 解析器时，仅在本次 Python 内存命名空间将尺寸上限设为 1024，以读取既有 768px 裁片；没有修改 binder、原始 PNG 或正式 capture 上限。

| 点位角色 | 唯一坐标数 | 核对规则与证据 |
| --- | --- | --- |
| `known_hole` | 63 | 第 22.8.3 节六段中的全部像素逐一 alpha=0；上部 59 点同时由 `qg-roi-08.png` 和 `upper-coverage-step.png` 核对，右下 4 点来自 `qg-roi-07.png` |
| `opaque_control` | 24 | 每段中点正上、正下各一像素，左右端点外各一像素；四向对照逐一 alpha=65535，按 canvas 坐标去重 |
| `outside_coverage_control` | 2 | 从实际画布边界连通的 alpha=0 区域选取，保留整段透明连接路径；不将任意 alpha=0 点自动标为合法外覆盖 |
| 合计 / 硬上限 | **89 / 128** | label 与 canvas 坐标唯一；重复坐标、冲突角色或超过上限均拒绝，不静默截断 |

两个负对照分别为 canvas **`[15865,6468]`** 和 **`[15667,6688]`**。前者在 `right-edge-coverage.png` 中沿同一行到实际右边界 `[15896,6468]`，后者在 `qg-roi-07.png` 中沿同一列到实际下边界 `[15667,6719]`；各有连续 **32 个 alpha=0 像素**作为连接证据。这里的角色描述既有输出的外覆盖拓扑，不等于已经独立证明所有 RAW 在这些世界坐标均无有效支撑。

输入 schema 为 1，reference canvas origin 为 **`[-108,-49]`**；世界坐标由 `canvas + reference_canvas_origin` 得到。实际 compositor 若发生裁剪或原点改变，须再根据其记录的原点映射，不能直接把旧 canvas 坐标当作新图坐标。输入文件 [`points-input.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v21-holes/points-input.json) 的 SHA-256 为 **`6ca08aad4c67c719d3e38cdc5808c6131d504d8d65cf9b0c3aae8706f7bb68bb`**。[`points-evidence.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v21-holes/points-evidence.json) 保存逐点 canvas/world/local 坐标、alpha、PNG 路径和 SHA、段落/四向关系，以及负对照的边界连接路径。

### 24.2 默认关闭的只读 trace：区分实际写入与辅助来源诊断

[`native_point_trace.rs`](/Users/dp/Desktop/raw-editor/src-tauri/src/panorama_utils/stitching/native_point_trace.rs) 及 compositor 接入点均受 `cfg(test)` 约束。普通生产构建不包含该诊断模块；测试构建只有同时显式设置 `RAW_EDITOR_NATIVE_POINT_TRACE_INPUT` 与 `RAW_EDITOR_NATIVE_POINT_TRACE_DIR` 才启用，默认关闭。关闭时不创建诊断目录、不读取输入文件或分配点位 ledger。没有更改生产取样、owner 选择、coverage 写入、配准或 QG 阈值，也未改变 Stack_Report schema。

每个 station 记录世界坐标到工作 canvas/tile 的映射、bbox 与 tile 范围、实际 owner 选择、sampler 是否调用、返回值、黑色过滤和亮度/质量拒绝路径，以及写入前后的 coverage、station owner 和 RGB。最终另存 compositor 完成裁剪后的尺寸、原点、89 点 coverage/owner/RGB。记录只覆盖请求点，不创建全图 trace plane。

辅助来源部分复用当时已驻留的 originals，检查局部 owner 候选、逆 refinement、source sampling、source footprint、prepared-image 边界及现成来源取样值；**不额外调用 source loader 或 RAW 解码器**。但开启后仍会增加这些点的几何计算、内存取样和 JSON I/O，不能称为零成本。`actual` 是生产路径的实际控制流记录；`native_candidates` 的 3×3/旧 4×4 owner 与额外来源检查是诊断计算，不能把它们的数量当作实际 sampler 调用次数，或把诊断候选存在当作该来源已写入最终像素。

三种 owner 语义必须分开：compositor 的 `station_owner` 是 **station 顺序下标 + 1**；局部来源诊断的 `local_owner` 是该 station 的 legend 下标 + 1；最终 QG source capture 中的全局 RAW owner 才通过其 sources 表指向输入文件。不能将 compositor owner 数字直接解释为某张 NEF，亦不能用 station owner 覆盖数替代第 23.8 节口径的最终 RAW census。

每次 trace 保存到 `point-trace/process-PID/iteration-NNNN/trace.json`，对应 `latest.json` 只声明 `running`、`compositor_complete` 或 `failed`。新尝试会先使旧 latest 失效，启动/完成 I/O 失败及未完成退出均不得继续沿用旧 complete。原始 **`final_iteration=null`、`offline_binding_allowed=false`** 保留为候选状态：compositor 完成不代表后续 QG/repair 和 TIFF 已完成，最终关系由离线绑定建立，不回写原始 trace。

### 24.3 已完成的验证与正式运行配置

4 项 Rust 定向测试覆盖输入与点数边界、局部 owner/来源支撑诊断、开关前后像素保持及真实黑色拒绝记录，以及后续失败使旧完成迭代失效。它们已经计入上述 999 项通过的全库回归，不重复相加。脚本准备检查另见 [`preparation-checks.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v21-holes/preparation-checks.json)；该文件保留生成时的 `prepared_not_run`，不追溯改写为后来的正式运行结果。

[`run-trace.py`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v21-holes/run-trace.py) 继续使用同一组 `DSC_6479.NEF`–`DSC_6487.NEF`、`scale=1`、coverage alpha、2,400,000,000 B source cache、4 GiB native cache 与原有 RSS/QG 门禁。启动前核对固定 binary SHA，清除继承的全部 `RAW_EDITOR_*` 后显式配置本次变量，使用互斥锁和不允许复用的运行目录，并保存输入/evidence 的运行内副本。native wrapper 单列 Rust 子进程 wall 与 OS maximum RSS，不重新编译二进制。

本次同时启用既有 **8 ROI** source-visual capture，用于最终 QG 原点/来源与 TIFF 的绑定；没有启用 final-world-owner 或 gradient oracle。该诊断运行与第 23 节的性能冷/热分开，即使 RAW decode 或 wall 数值恰好相同，也不能把包含 trace/capture I/O 的时间并入性能收益比较。

正式 trace 已完成，[`trace/execution.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v21-holes/trace/execution.json) 的 exit code 为 0，固定 binary、运行内点位输入及 alpha evidence 副本均未改变。相对 v20 的 [`comparison-to-v20.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v21-holes/comparison-to-v20.json) 中 14 项报告/几何/来源/QG/owner/output/budget/TIFF 检查全部 pass、changed leaf 均为 0。TIFF 仍为 **15897×6720、854,628,326 B**，SHA-256 **`e16fd14781d86258cdeb7912abc43a3b230fa9935ec1488e0c1acda68901b605`**，与 v20 相同；这验证诊断没有改变本次输出，不表示已消除孔洞。

正式 QG 仍为 pass、8 ROI，MTF 仍是 512 候选中的 1 个可测而保持 N/A，boundary 仍为 372 候选中的 0 个可测而保持 N/A；degradation 仍有 4 条 grouping motion 与 6 条 low-sharpness。RAW decode=67、fusion=6。诊断 Rust 子进程 wall 为 364.398869s、OS maximum RSS 为 11,925,143,552 B，计时含 trace/capture 工作，仅归档观测，不据此推导新的速度或内存收益。

### 24.4 离线 trace→TIFF 绑定规则与真实结果

[`bind-point-trace.py`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v21-holes/bind-point-trace.py) 以同次成功运行的 source-visual binder 为前置：要求 release/debug assertions=false、完整 8 窗绑定、execution/report/log/capture 证据哈希一致，并核对 trace 与最终 QG 属于同一进程、最新 compositor 候选没有后续失败或未完成迭代。compositor 与 QG capture 的迭代计数独立，分别记录，不假定编号相同。

TIFF SHA 沿用该次 source-visual binder 的成功记录，先核对 inode、大小、mtime、ctime 快照仍一致，再用既有未压缩 TIFF strip reader 对每点 seek 读取 1×1 RGBA16；完成后重新核对 stat 和小证据文件，避免重新解码或复制全图。89 点的像素负载读取为 712 B，另有 TIFF header 与证据文件读取；不能把这个数字当作脚本全部 I/O。

RGB 从 JSON 恢复为 f32，沿用 v20 binder 的生产量化：clamp 到 `[0,1]`、binary32 乘 65535、半值远离零舍入到 u16；不再次施加 gamma、色调拟合或插值。coverage bool 对应 alpha 0/65535，未覆盖点的未预乘 RGB 也逐通道比较。缺字段、无法恢复的 RGB、坐标/尺寸不一致、旧 complete 或跨目录 symlink 均不能产生通过结果。小合成核验共 **16 项通过**，包括 2 像素 TIFF 的实际 seek、RGB/alpha mismatch、station 统计与 owner 语义、缺字段、后续失败、路径与 TIFF stat 变化；记录见 [`point-binding-tool-checks.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v21-holes/point-binding-tool-checks.json)。这不是正式 89 点的绑定结果。

| 真实诊断验收项 | 当前状态 | 已归档证据 / 限制 |
| --- | --- | --- |
| 9 NEF 完整运行、binary/点位副本不变 | exit 0，完整输出与 v20 等价 | execution、Rust log、Stack_Report、14 项比较 |
| 最终 QG 与 8 ROI source capture→TIFF | `bound`，8 窗 exact | `trace/source-visual-binding/manifest.json` |
| 最新 compositor 候选与 TIFF 的 89 点 RGB/alpha | `bound`，89/89 exact | `trace/point-trace-binding/summary.json` |
| 63 孔逐 station 实际拒绝原因 | 全部 `native_sample_returned_none` | 最近来源映射失败，同邻域存在有效替代；见第 24.5 节 |
| 24 opaque / 2 exterior controls | 24 点保持 alpha=65535，2 点保持 alpha=0 | 全部与原始角色及 trace 一致，无新增灰度 alpha |
| 修复和孔洞复验 | **未验收；63 孔仍在** | 后续共享几何 resolver 候选须单独冻结与验收 |

本次 source capture 与 compositor trace 均来自 `process-45969`，离线分别绑定到各自的 iteration 1；两套计数独立，原始 `final_iteration=null` 未修改。实际工作 canvas 为 **15931×6732、origin `[-111,-53]`**，裁剪后为 **15897×6720、origin `[-108,-49]`**，因此这些点的工作 canvas 坐标为最终 canvas 加 `[3,4]`。绑定使用世界坐标和各自原点核对，没有把裁剪前后坐标直接混用。

[`source-visual-binding/manifest.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v21-holes/trace/source-visual-binding/manifest.json) 的 8 窗精确绑定与 [`point-trace-binding/summary.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v21-holes/trace/point-trace-binding/summary.json) 的 89 点精确绑定是不同范围的证据。后者全部角色保持原始 alpha：63 孔与 2 外覆盖负对照的最终 compositor owner 均为 0，24 opaque 点为 compositor station owner 1。最终 QG 全局 RAW census 也保持 v20 的 97,858,218 个有效覆盖像素、covered owner-zero=0，6484 贡献 9,462,041 像素；该全局计数不把 63 个透明孔纳入有效覆盖分母。

`bound` 只证明指定样本与最终 TIFF 一致。本次孔仍透明且绑定 exact，并不矛盾；完整原生质量/最佳焦点验收仍未通过。

### 24.5 独立逐点归因：无效最近来源阻断同邻域的有效替代来源

[`summarize-holes.py`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v21-holes/summarize-holes.py) 直接读取绑定所引用的原始 trace JSON，核对其 SHA、latest、execution 与报告身份，再逐点重算；没有只复制 binder 的预汇总，也没有重新加载 RAW 或 TIFF。完整逐点、六段各自四向邻点及来源坐标归档在 [`hole-diagnosis.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v21-holes/hole-diagnosis.json)，精简机器可读记录为 [`sleeping-9-v21-holes-diagnostic-acceptance.json`](benchmarks/sleeping-9-v21-holes-diagnostic-acceptance.json)。

这 89 点均位于 **compositor station owner 1 / topology station 2** 的 bbox 和 tile 范围内。该站实际记录为 63 孔与 2 外覆盖点 `native_sample_returned_none`、24 opaque 点 `mask_written`；另外两个 station（topology 0、1）的 178 条点记录全部为 `outside_station_bbox`。不能因此把编号 1 解释为全局 RAW owner 1，或说 station 0 覆盖了这些孔。

63 孔均已通过 station owner 选择并实际调用 native originals sampler，返回 None 后没有写入 mask。其最近 3×3 局部 owner 均为 **4（legend index 3，DSC_6482.NEF）**；对应 source diagnosis 为 `source_mapping_returned_none`、`global_source_supported=false`、local band count=0。63 个未过滤的全局源坐标都仍在 prepared-image **8312×5542** 的矩形范围内，失败边界来自 source footprint 支撑，不能简化为图像矩形越界。

同时，每点的同一 3×3 owner 邻域都包含另一来源：逆映射成功、render bounds 与 source footprint 有效、诊断 RGB 有限且大于现有黑色阈值。下表按这些既有候选中的最近距离统计；“有效替代”是本次只读诊断观察，**没有在冻结版本中真正被重新选择或写入**。

| 区域 | 失败最近来源 / local owner | 同 3×3 最近有效替代 / local owner | 点数 |
| --- | --- | --- | --- |
| 上部五段 | DSC_6482.NEF / 4 | DSC_6480.NEF / 3 | **59**（4+10+14+20+11） |
| 右下细段 | DSC_6482.NEF / 4 | DSC_6479.NEF / 1 | **4** |

两个具体邻点例子保留了原始距离和几何链，便于后续回归复现。上部 canvas `[9345,522]` 映射到 tile `[2410.503081,71.569317]`：最近 owner 格 `[2411,72]` 的距离平方为 `0.432416130`，选中 6482 后得到未受支持的源点 `[2113.864869,17.727473]`；同邻域 owner 格 `[2411,71]` 距离平方稍大，为 `0.571050584`，其 6480 源点 `[2334.683751,47.591898]` 支撑有效，诊断 RGB 约为 `[0.442529,0.400141,0.318131]`。这说明离散 owner 格上的最近标签不保证当前连续目标经过该 RAW 的 refinement/逆映射后仍受支持。

右下 canvas `[15645,6459]` 映射到 tile `[8352.015204,5611.411349]`：旧格 `[8352,5611]` 的距离平方 `0.169439326` 对应 6482，其源点 `[8067.561959,5535.892189]` 在矩形内但 footprint 无效；格 `[8352,5612]` 的距离平方 `0.346740954` 对应 6479，其源点 `[8293.015204,5291.411349]` 支撑有效，RGB 约为 `[0.621106,0.602640,0.637172]`。

四向 opaque 对照并非全由替代 RAW 提供：24 点中 **12 点仍选 6482、10 点选 6480、2 点选 6479**，全部实际写入。这排除了“整张 6482 不可用”的解释。例如上部孔左邻 `[9344,522]` 与下方控制 `[9346,523]` 均由 6482 正常写入，对应有效源点 `[2112.903440,17.746162]`、`[2114.810933,18.657988]`；右下控制 `[15646,6458]` 与 `[15649,6459]` 也由 6482 正常写入。失效集中在当前坐标的支撑判断与 owner 选择组合，不能整帧剔除处理。

两个 exterior controls 则都没有 3×3 有效替代：九个局部 owner 全为 0，nearest winner 为 null、source diagnostics 为空；native sampler 返回 None 与保持透明一致。这是后续修复必须保留的负对照，不能通过无条件补 alpha 或扩大到无证据的远处来源填补。

**黑色过滤不是本次 63 孔的实际退出分支。** 它们在 sample=None 后已退出，actual ledger 中没有执行 `black_filter_le_1e_minus_6`；降低黑色阈值无法修复这条路径。直接缺陷是先按离散 owner 邻近距离选择单一来源，该来源的连续几何支撑失败后直接返回 None，没有继续考虑同一有界邻域内已存在的有效候选。这个结论限定于已请求和绑定的 63 点，不冒充全图孔洞普查或全部潜在缺陷的唯一原因。

### 24.6 后续修复边界与尚未完成的验收

下一候选在共享几何 resolver 中统一 native render 与 QG 来源逆链的有效性判断，保留有界邻域和确定性距离顺序，在候选的真实映射/footprint/bounds 合法后再采用；同时保留完全无有效候选时的透明输出。当前已启动最小修复和定向回归工作，但它不属于上述 `83a74…1aa0` 冻结二进制，须另设目录、冻结 SHA 和完整验收，不能覆盖本次 raw trace 或将诊断候选回写成已修复结果。

修复验收须重新核对 63 孔、24 opaque 与 2 exterior controls、来源身份/坐标与 final owner、原 QG 门禁及 N/A/degraded 状态、生产 TIFF 契约和局部原生视觉；变更像素不能再要求整 TIFF 与旧含孔版本 SHA 相同，而应解释预期变化并检查旁侧区域。软印章、条带与全部焦点候选的最佳选择仍是独立未完成项。第 23 节已完成的性能结论继续保留，当前阶段仍不能标记总体任务或完整画质验收完成。

## 25. v22：按几何支撑回退 native owner（63 孔恢复覆盖，严格对照门禁未通过）

本阶段针对第 24 节定位的退出路径实现最小修复：最近的局部 owner 在当前连续目标坐标上不受该 RAW 支撑时，继续检查同一 3×3 邻域内其余来源。native RGB 取样与最终 RAW owner 证据共用几何选择规则，避免图像使用替代来源而 QG 仍回指原无效来源。真实 9 NEF 的冷跑、热跑、独立 capture 和全 TIFF 差分已完成；63 个已知孔均已恢复不透明，并具有实际来源及最终 QG owner 证据。**但 24 个原样不透明对照中有 2 点的最终 RGBA16 改变，预先设定的严格 repair 门禁仍为 failed；不得写成 v22 整体验收通过。**

候选验收使用独立目录 [`sleeping-9-native-v22-owner-support`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v22-owner-support)，固定二进制路径为 `/private/tmp/raw-editor-v22-owner-support-release-binary`，SHA-256 **`29f344a83bc403ae0f08a5ac88d65efb2641c0505e2db0fe0a1a597c7d74d359`**。[`build.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v22-owner-support/build.json) 保存源码清单与回归结果，精简验收记录见 [`sleeping-9-v22-owner-support-acceptance.json`](benchmarks/sleeping-9-v22-owner-support-acceptance.json)。第 24 节的冻结诊断、含孔 TIFF、89 点原始 trace 与 alpha evidence 保持不变。

### 25.1 共享纯几何 resolver：有限候选、真实支撑、确定性选择

新增 [`native_owner_geometry.rs`](/Users/dp/Desktop/raw-editor/src-tauri/src/panorama_utils/stitching/native_owner_geometry.rs)，将 native RGB 和来源证据需要的选择统一为 `resolve_native_owner_geometry`。其处理顺序如下：

1. 仍以目标 tile 坐标的 rounded centre 为中心，读取原有 **3×3** owner stencil；忽略越界格和 owner=0，不扩大搜索范围。
2. 对最多九个格中的相同 owner 去重，每个来源保留最近证据格。按原距离平方运算排序，距离相同保留原 row-major 顺序，不改为 owner ID 顺序；候选存于固定栈数组。
3. 逐候选执行其既有 inverse refinement、native sampling origin 与 `source_point` 映射，检查原 source footprint、metadata bounds 和实际 prepared buffer 的排他上界。返回的源坐标不 round、不 clamp。
4. 采用第一个几何有效的来源，返回 local owner、证据格、prepared source point 和 `used_fallback`。原最近来源合法时仍保留原选择；全部无效则返回 None，保留无支撑区域的透明结果。

来源选择不读取 RGB、sharpness 或插值结果，也不按亮度改选来源。owner 对应的 source、sampling metadata 和精确 prepared-buffer dimensions 必须在像素循环前完成校验；缺失元数据是 payload 错误，不能借“回退”掩盖缺失输入。该函数复用既有映射和 footprint，不引入新的几何估计、配准阈值或远处来源。

### 25.2 RGB、reverse owner 与 QG 来源链使用同一选择

native renderer 使用共享 resolver 返回的同一 prepared source point 执行原高质量插值。reverse/source evidence 使用相同的不可变 sampling metadata、prepared dimensions 和选择规则，从而将最终全局 RAW owner 指向实际取样来源。旧 **4×4** 邻域现在仅保留作 coverage 诊断，不能再用它推导与 native render 的 3×3 选择不一致的最终 RAW 归属。

黑色阈值、亮度/质量拒绝路径、配准、station 布局、density、QG 阈值及 legacy 取样路径保持原约束。此修复没有放宽 `black_filter_le_1e_minus_6`，也没有以补 alpha 或 RGB 黑色判断替代几何支撑；第 24 节已证实的 63 孔在旧版 sample=None 后退出，尚未进入黑色过滤。

仍须区分三套编号：compositor station owner、station legend 内的 local owner，以及最终 QG 的全局 RAW owner。新的实际 trace 在产生 RGB 的**同一次** `sample_with_geometry` 调用中保存 `actual.native_owner_resolution`，包含 local owner、legend index、证据格、prepared point、fallback 标志、source ID/path 和 sample RGB；旧 `native_candidates` 继续描述原 nearest 规则的诊断候选，不能将诊断 replay 当作实际写入证据。

### 25.3 同一点的最终 QG RAW owner 证据

既有 8 窗 source capture 没有覆盖全部已知孔，因此新增 [`source_visual_point_evidence.rs`](/Users/dp/Desktop/raw-editor/src-tauri/src/panorama_utils/stack_pipeline/quality_gate_runner/source_visual_point_evidence.rs)：在显式启用的测试诊断运行中，对同一份 89 点输入记录当次不可变 QG input 的实际 ownership、coverage、输出 RGB f32/bits、来源路径、station ID 与 prepared source 坐标。来源取自 `input.ownership` 的真实像素值，不从全图 census、ROI 中心 owner 或 compositor station ID 推测。

逐点同时保留输入定义的 requested world 与实际 QG canvas/world 坐标，按现有完整逆链得到 source point；无法对齐整数网格、缺来源元数据、坐标非有限或越界均进入 capture error，不能静默写成 owner=0。未覆盖点保留观察到的实际 owner，source coordinate 为 null。该 ledger 复用现成 output/ownership/几何信息，**不新增 source loader 调用或 RAW 解码**。

文件写入 `source-visual/process-PID/iteration-NNNN/point-evidence.json`；父 metadata 记录 path、SHA、bytes、input SHA、点数和 iteration。模块受 `cfg(test)` 与显式 capture 配置控制，默认关闭。只有整次运行成功、父 latest 为最新 complete 且无 capture errors、最终 TIFF 已绑定、各文件身份一致后，离线工具才将该候选与最终输出关联。原始 `final_iteration=null` 不改写；compositor 与 QG 的迭代编号分别核对，不假设两者相同。

### 25.4 当前已完成的验证范围

生产 `cargo check` 已通过。debug 定向结果为：native owner **10 项**（含 render→reverse→QG 综合核对）、tile payload **7 项**、source visual **13 项**（其中包含 point ledger **6 项**）、native point trace **4 项**均通过。冻结 release 全库为 **1016 passed / 0 failed / 40 ignored，158.70s**。各测试过滤组不与全库结果重复相加，point ledger 也不重复计数。

离线工具的准备与纯小合成检查见 [`tool-preparation.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v22-owner-support/tool-preparation.json)：runner、TIFF hash、点位 binder、全 TIFF 比较、repair verifier 与质量/性能比较器六组自检通过。合成例覆盖 endian、strip 内行偏移及最后不足一 strip、RGB/alpha 差异、旧 coverage 丢失、来源/坐标/迭代不一致、后续失败、错误基线与证据漂移。该文件保留准备时的 `prepared_real_v22_acceptance_pending`，不追溯改写；正式结果以下述各次运行及绑定记录为准。

### 25.5 正式验收顺序与不可省略的条件

已按冻结 release binary SHA 与源码清单、全库回归、cold、hot、独立 capture 的顺序完成运行，三次均 exit 0。继续使用 `DSC_6479.NEF`–`DSC_6487.NEF`、scale=1、coverage alpha、2,400,000,000 B source cache、4 GiB native cache 和原 25,769,803,776 B RSS 门禁。runner 清理继承的 `RAW_EDITOR_*`、检查固定 binary SHA、使用互斥与独占目录，并验证前一阶段完成，拒绝覆盖旧证据。

cold/hot 均关闭 point trace 和 source capture；第三次 capture 同时启用 8 ROI source capture、冻结 v21 输入的 89 点 trace 与最终 QG 点位 ledger。capture 的几何诊断、JSON 与图像 I/O 单列，不能混入性能样本。冷/热均为新进程，不宣称已清空 OS 文件缓存；一对样本只能报告实测差异，不能推导统计置信。

| 验收层 | 必须核对的条件 | 当前结果 |
| --- | --- | --- |
| 63 个已知孔 | 每点最终 alpha=65535、实际 `mask_written`、`used_fallback=true`，来源为原同 3×3 内几何有效候选；实际 RGB、source path/point 与最终 QG 全局 RAW owner 逐点对应 | **63/63 有界来源与覆盖证据完成**；不覆盖其他验收项 |
| 24 opaque controls | 与冻结 v21 的完整 RGBA16 保持一致，原 source/local owner 不变，不发生 fallback | **严格 RGBA16 条件 failed：22/24 exact，2 点改变**；24 点来源、坐标、实际 sample 与首次写入均 exact，无 fallback |
| 2 exterior controls | 保持透明、无实际写入、无来源坐标，compositor owner 和最终全局 RAW owner 保持 0 | 2/2 保持原状态 |
| 89 点与 8 ROI 输出身份 | 同次成功 release 运行、最新 compositor/QG 迭代无后续失败；89 点 RGB/alpha 和 8 个原生 ROI 均逐像素绑定实际 TIFF | 两套 binder 均 `bound`，89/89 点与 8/8 窗 exact |
| 同点来源闭环 | 87 个 covered 点核对 actual render source 与最终 QG owner；prepared XY 仅容许明确记录的 `1e-7` native pixel 数值误差，不调整坐标以制造一致 | 87/87 完成；63 孔的实测最大 XY 误差为 0 |
| 全 TIFF 差分 | 流式比较冻结 v21 与候选所有像素；旧 covered 丢失=0、新增非二值 alpha=0、新增 covered 至少 63；保留全部 RGB/alpha 变化数量、范围与分块定位 | 新增 covered **553**、丢失 **0**、非二值 alpha **0**；另有 **12,194** 个旧 covered 像素 RGB 改变，须继续审核 |
| 新版冷/热/capture 一致性 | 同一冻结 binary；除计时、运行身份与 RSS 采样等运行项外业务报告一致，三次 TIFF SHA 相同；SHA 与读取前后完整 stat、execution/report 身份关联 | cold→hot、hot→capture 均 pass；不是旧版像素不变的声明 |
| 质量、性能与视觉 | 现有 QG 策略不变，保留 measured/N/A/degraded；核对 RAW、fusion、缓存、wall、RSS，并查看孔及全图差分涉及的原生区域 | QG pass、MTF/boundary N/A、degraded 保留；性能已记录；24 个变化/上下文窗口及 8 对来源窗口已实际查看，严格门禁与完整画质仍未通过 |

旧版与新版预期有像素变化，**不要求 v22 TIFF SHA 等于含孔 v21**。`bind-point-trace.py` 只判定 trace 是否对应实际 TIFF，因此 known-hole 的 alpha 改变不会被错误当作绑定失败；是否完成修复由 [`verify-repair.py`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v22-owner-support/verify-repair.py) 的角色、来源与全图条件另外判断。全图变化也不能预设“只有 63 像素”：共享 resolver 可能覆盖更多同类失效点，必须通过 [`compare-whole-tiff.py`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v22-owner-support/compare-whole-tiff.py) 统计和解释所有变化。该比较按 TIFF strip/行读取，保留有限行块及 changed-tile 统计，不一次解码两张完整 raster。

QG 按现有 schema 中的 criterion 名称和策略逐项核对。冻结基线有 `local_scale_median`、`local_scale_pixel_ratio`、`effective_pixel_count`、`mtf50_normalized`、`gradient_energy_normalized`、`noise_sigma_ratio`、`roi_delta_e00`、`boundary_stroke_alignment`、`owner_sharpness_coverage` 九项；比较器检查其 threshold、上下界、diagnostic、selection mode 和 reference ROI limit，而不以“数量为 9”判定通过。NEF 输入数量也单独核对。**v22 的 MTF eligible/measurable 已从基线 1 降至 0/512，仍为 N/A；boundary 为 0/372、仍为 N/A**，不能照搬旧版 MTF 数字或宣称测量完全相同。grouping/low-sharpness degradation 继续独立呈现，不用 QG pass 代替全图画质验收。

[`comparison-final.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v22-owner-support/comparison-final.json) 中，输入、分组、topology、配准、closure、residual warp、density、fusion、QG/RSS 策略的跨版本核对均 pass；它允许并完整记录像素、实测 QG、tone 和 degradation 的变化。该比较器的 pass 表示这些条件和新版重复运行一致，**不包含、也不能覆盖严格 repair verifier 的 failed**。性能只比较 v20 cold/hot 与 v22 对应阶段，以及 v22 自身冷/热；capture 不参与收益计算。

### 25.6 真实绑定、全图差分与严格门禁失败

三次 v22 输出均为 **15897×6720、854,628,326 B、RGBA16 unassociated alpha**，world origin 为 `[-108,-49]`，TIFF SHA-256 均为 **`718a903dcd0bf102fd7ea7f2be652168ad972f18bc540f5854225ee2f1bb47e3`**。它不同于第 24 节含孔基线 `e16f…b605`；这是实际像素改变，不能写成跨版本 TIFF 等价。capture 来自 process **69962**，source capture 与 compositor trace 分别绑定到各自 iteration 1，原始 `final_iteration=null` 保持不变。

[`verify-repair-components.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v22-owner-support/verify-repair-components.json) 在保留原严格 verifier 失败记录的基础上，将各项证据单列：63 孔全部实际 fallback 写入，原失败来源 6482 的替代为 **6480×59、6479×4**；87 个 covered 点的 actual source 与最终 QG 全局 RAW owner 对应；24 个 opaque 对照的来源、prepared coordinate、实际 source sample 和首次写入逐位不变；2 个 exterior 对照仍透明。该文件状态明确为 `component_evidence_complete_strict_final_RGBA_failed`，不是重新定义成功条件。

[`repair-verification/summary.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v22-owner-support/repair-verification/summary.json) 的失败原因为 `opaque control RGBA changed from frozen v21 TIFF`。两个具体失败点为 canvas `[9344,522]`，R 从 25436 变为 25437；以及 `[9393,524]`，R 从 26341 变为 26342、B 从 20299 变为 20300；两点 G 与 alpha 不变。**差异小也不等于原样条件通过**，没有追加“允许 1 LSB”的后验阈值，没有覆盖旧失败产物。

[`independent-control-stage-analysis.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v22-owner-support/independent-control-stage-analysis.json) 将范围限定于 24 个请求对照：actual sample 与首次写入均逐位相同，但最终浮点 RGB 仅 7/24 逐位相同，最终 RGBA16 为 22/24 相同。由此可确定这些点的变化出现在首次写入之后，不能归咎于 resolver 在对照点改选 RAW。两份报告中 tone 有效样本按 station 2/0/1 分别增加 **13/2/2**；代码调用链和拟合参数变化支持“新增支撑影响后续 `supported_spatial_affine` 色调处理”的解释。现有 trace 未单独捕获 revisit 后、tone 前的逐点 RGB，因此这是代码与报告相互印证的归因，不能冒充逐阶段隔离实测，也不能直接解释全图所有 12,194 点。

[`whole-tiff-diff/summary.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v22-owner-support/whole-tiff-diff/summary.json) 遍历全部 **106,827,840** 个像素：RGBA 改变 **12,747** 个，其中新增 covered **553** 个、旧 covered RGB 改变 **12,194** 个；旧 covered 丢失 0、仍未覆盖区域的 RGB 改变 0、两版非二值 alpha 均为 0。有效覆盖从 **97,858,218** 增至 **97,858,771**，新增 553 中除已核对的 63 孔外还有 **490** 点，不能假定它们都等同于这 63 点的来源验收。变化分布在 **29 个 256px tile**，旧 covered RGB 变化的 bbox 为 `[6933,302]`–`[9507,4814]`。

进一步的 [`covered-delta-analysis.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v22-owner-support/covered-delta-analysis.json) 对完整差分涉及的 tile 重算，并核对数量：旧 covered 像素的单像素最大通道差为 1/2/3/4 个 u16 单位时，分别有 **11,988/163/42/1** 点；最大为 **4/65535**。这是测量分布，不是新增容差。全图最大 RGB 差 65535 包含原透明变为有效内容的点，不能误用为旧 covered 区域的最大色差。

最终 QG owner census 覆盖 **97,858,771** 像素，covered owner-zero 为 0，九张 RAW 均有正贡献；6484 的最终贡献为 **9,462,219** 像素。该计数来自真实最终 QG owner plane，与 station 内 fusion owner 数不同；既不能用 census 替代逐点来源核验，也不能据此证明 best-focus 选择。

### 25.7 真实 QG 与冷/热性能记录

三次运行均为 QG pass、8 ROI，其中 station-owner evidence 8、raw-owner evidence ROI 0、unresolved pixel 0；另行捕获的 89 点全局 RAW ledger 不更改该报告 ROI 口径。现有九项中七项 pass，MTF 与 boundary 为 N/A。MTF 的 512 候选全部被内容条件拒绝，boundary 的 372 候选均为低对比拒绝；noise 在 46 个已访问候选中有 24 个可测、22 个非平坦拒绝，另有 210 个未访问候选。degradation 仍为 **degraded**，保留 4 条 grouping motion 与 6 条 low-sharpness 记录。

ROI 位置和来源已改变，不能声称测量原样。QG reference 从 **33 降至 32**、source-cache QG hits 从 **55 降至 51**，QG owner-source decode 从 **13 增至 14**；总 RAW decode 仍为 **67**。ROI cache hits 仍为 0，misses 从 33 降至 32。该差异伴随 MTF 可测数及候选选择改变，不应归为新的 cache 命中收益。

| 指标 | v20 cold | v22 cold | v20 hot | v22 hot |
| --- | ---: | ---: | ---: | ---: |
| Rust 子进程 wall（s） | 369.421936 | **378.733059** | 369.730270 | **364.199523** |
| 外层 execution wall（s） | 370.144676 | 379.426737 | 369.951530 | 364.370507 |
| report render（s） | 351.760724 | 361.240029 | 349.263916 | 347.465893 |
| native fusion wrapper 累计（s） | 213.370829 | 218.584306 | 212.483939 | 211.390811 |
| fusion 次数 / RAW decode | 6 / 67 | **6 / 67** | 6 / 67 | **6 / 67** |
| QG reference resampling（s） | 19.282266 | 18.805893 | 19.643608 | 18.462814 |
| OS maximum RSS（B） | 12,132,499,456 | **12,115,066,880** | 11,792,138,240 | **12,432,506,880** |

相对 v20 对应阶段，v22 cold 的 Rust wall **+2.52%**、OS RSS **−0.14%**，hot 的 Rust wall **−1.50%**、OS RSS **+5.43%**。本次没有进一步减少 fusion 或 RAW 解码，也没有稳定的速度/内存收益结论；单对样本、未 purge OS cache 及 QG 选取改变均限制解释。原有 source/native cache 预算和 RSS 门禁未放宽。

独立 capture 的 Rust wall 为 **367.359404s**、OS maximum RSS 为 **12,368,412,672 B**，RAW=67、fusion=6，单独归档，不计入上述性能比较。closure 仍 converged，median/p95 为 **0.488082502/1.287012289px**，density factor **1.095703125**，direct-pair p95 输出尺度 **1.499558378px**；其采样 Jacobian 契约没有升级为全像素几何证明。

### 25.8 已完成的有界视觉复核与后续验收决策

本次已有 12,194 个旧 covered RGB 像素改变，不能整体复用第 22 节旧 30 窗的视觉结论。原生导出工具复用 upper/right 两个 768px 孔洞上下文窗口，并将全部 29 个 changed tile 归并为 **22 个 512px 窗口**；变化窗口硬上限 24，实际导出 **24 窗（2 固定 + 22 变化）**。[`visual/manifest.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v22-owner-support/visual/manifest.json) 记录逐 strip 原生裁片、无 resize、RGBA16 与来源 sRGB ICC，以及所有 changed tile 均完整处于导出窗口内、遗漏分组为 0。窗口可能重叠，该覆盖声明不是全图每个像素都经过人工判断。

[`root-review.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v22-owner-support/visual/root-review.json) 实际查看两个固定窗、changed-01–06 与 changed-19–22 共 **12 窗**；[`independent-review.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v22-owner-support/visual/independent-review.json) 实际查看 changed-07–18 共 **12 窗**。按名称、路径、PNG SHA 与导出 manifest 核对，两组交集为 0，合计 **24/24 导出窗口均已实际查看**。原孔位置的图像内容连续，所阅窗口未发现可确认的新增孤立缺口、重复笔画或明显矩形接缝；合法外覆盖透明轮廓仍在。部分红印软边、横向织纹/亮带、近白高亮和细彩边仍可见，不能写成已解决或处处清晰。≤4/65535 的数值差异由精确比较建立，不能宣称屏幕观察已经分辨并认可这些差异。

另外，[`source-visual-review.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v22-owner-support/capture/source-visual-binding/source-visual-review.json) 已对 3 个 station 的 **8 对原生 256px output/reference、共 16 张 PNG**逐一实际查看。每窗可能包含多个真实 RAW owner；reference 是既有 QG 选中 owner、经过现有 tone 的对照，不是全部焦点候选中的最佳图。所阅输出与参考的笔画/纹理位置相符，未辨出新增明显重影、断纹或接缝；红笔软边和部分织纹柔和在参考中同样存在。ROI03/05/06 的浮点 output/reference 相同，其余有差异；最大单通道差位于 ROI07，为 `0.003050297498703003`。这与 8 窗 output→最终 TIFF 均 exact 属于不同的比较关系，不能相互替代。ROI00/01 等窗口有重叠，8×65536 不是独立唯一像素数。

导出 manifest 保留其生成时的 `exported_pending_visual_review` 与原始空 reviewed 列表，raw capture 也保持原样；后续实际查看由上述独立 review 文件和本节记录，不回写旧产物。此次文档复核绑定的 manifest SHA 为 `69648562836f1cc44da6d71e640e50531a20941c318468f1397ae265b8b827dd`，root/independent review SHA 分别为 `366fc0c70853f5691f60cc5b45a86477a2a601fbdd2d626daa0989f56a12edd9`、`261981effea749136177c55fdcc69353e880bccc4ff0dbc6b48b61c9c2f65888`；来源视觉 review SHA 为 `6d8cae800834b85073067e7ae9b923dc2b3ede75846a3c53573ac9eedbb08377`。完整证据路径和 SHA 同步保存在本节 benchmark。

[`final-input-source-verification.json`](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v22-owner-support/final-input-source-verification.json) 已在运行后实际重核九张 NEF 字节、冻结 binary 和 **496 个**记录源码 SHA：全部一致，Rust 文件清单新增/删除均为空，fmt/diff 检查通过。该身份核对不解码 RAW，也不改变严格 repair failed 的事实。

下一阶段按以下顺序推进，保持当前原样门禁失败记录，不增加后验数值容差：

1. **先明确验收契约的层次并补隔离证据。** 当前已证实几何来源与首次写入保持、63 孔恢复覆盖；最终 RGBA16 原样仍失败。对 24 个对照及旧 covered 最大变化点 `[6949,4372]` 等有代表性的点，补默认关闭、只读的 revisit 后/tone 前、tone 后、量化前逐点观测，核对同一世界坐标、owner 和迭代。只有定位到具体阶段并满足预先明确的最终像素条件后，才能讨论 pixel identity pass；代码/报告印证和小差异本身不提供该结论。
2. **单独验收软印章与条带的最佳焦点。** 在已确认软边的红印和明显横向条带处固定原生世界坐标，对所有几何有效焦点候选导出同坐标、同尺度的来源片段及支撑 mask，比较当前 owner 与其他候选的笔画边缘、纤维连续性和噪声；不可只拿当前选中 owner 证明最佳焦点，也不可对无支撑区域强行比较。
3. **补齐额外变化的来源范围。** 当前 89 点 ledger 不足以逐点解释额外 490 个新增 covered 点和全部 12,194 个旧 covered RGB 变化。按变化分组选择有界证据，结合上述 tone 隔离结果解释变化来源，再决定是否需要扩大核验或调整实现。

有界视觉复核已完成；**严格 opaque-control 原样门禁、v22 整体验收和完整原生画质验收仍未通过**。软印章、条带、完整候选最佳选择及逐阶段像素保真继续作为明确的后续开发项。

### 25.9 后处理阶段观测接入（2026-11-03，尚未真实重跑）

已在默认关闭的 `cfg(test)` point trace 中增加五个同世界坐标快照：seam revisit 前、revisit 后、Tone_Harmonizer 前、Tone_Harmonizer 后、finishing 后/crop 前。每点记录 RGB 及 f32 bits、coverage、station owner、working canvas 和 iteration；只遍历原有限点清单，不加载 RAW、不复制整幅图像。结合既有 compositor return 与最终 QG/TIFF binding，可进一步定位后处理变化，但这些新增字段尚无真实 NEF 验收数据，不能据此改写 v22 的失败结论。

现有 native_point_trace 四项定向测试通过（含诊断开关前后像素一致与后续失败失效检查）。首次编译发现 metadata 可变/不可变借用冲突，已通过提前复制小型 working-canvas metadata 修复并重测。下一步仍需补阶段字段断言、冻结新 release，再执行真实逐点诊断；本节不复用旧 binary 的1016项回归作为新修改的验证。

### 25.10 v23 真实阶段诊断结果（2026-11-03）

阶段字段断言已加入既有诊断开关一致性测试，核对五阶段顺序、working canvas、89点同类固定清单的合成两点案例、coverage、station owner与RGB f32 bits；4项定向测试通过。新 release 编译完成，独立同组9 NEF、scale=1诊断运行 exit 0。产物位于 `output/acceptance/sleeping-9-native-v23-stage-trace`，build.json保存冻结binary和源码SHA，stage-analysis.json保存89点各阶段数据。此次未宣称新修改已通过全库回归。

实际运行外层457.179s、Rust子进程456.394s、OS RSS峰值10,250,829,824 B，包含诊断开销，不作为冷/热性能改善证据。8 ROI来源绑定完成；实际TIFF SHA仍为 `718a903dcd0bf102fd7ea7f2be652168ad972f18bc540f5854225ee2f1bb47e3`，与v22完全一致；全部89点compositor return也完全一致。

24个opaque对照逐阶段RGB变更计数为：seam revisit **0**，revisit后至tone前 **0**，Tone_Harmonizer **17**，后续finishing **0**。两个最终相差1色阶的对照均在新版tone阶段改变，其前后的seam/finishing未改变这些点。该数据测量的是v23内部阶段；冻结v21尚无相同阶段快照，因此不能冒充旧/新tone隔离A/B，也不能推广到未捕获的12,194点。此次保留原89点，尚未加入全图最大差异点[6949,4372]。严格最终RGBA原样门禁仍failed，下一步应补成对隔离证据及最大差异点，再推进软印章/条带全部有效来源比较。

### 25.11 v24：同二进制成对隔离证据（2026-11-03）

新增仅 `cfg(test)` 存在、默认关闭的 `RAW_EDITOR_DIAGNOSTIC_NATIVE_NEAREST_ONLY=1` 诊断模式：共享resolver判定需要fallback时，测试sampler返回None，复现此前最近owner无支撑即拒绝的行为；合法最近来源继续原样取样。该开关不进入生产构建、不调整QG条件。两次独立进程使用同一冻结release、相同9 NEF与scale=1、90点输入（原89点加全图最大差异点[6949,4372]），串行执行。10项owner定向测试及fmt/diff检查通过；没有把旧版本全库结果归于此修改。

产物为 `output/acceptance/sleeping-9-native-v24-paired-stage`：build.json保存冻结身份，capture/nearest分别保存运行，paired-stage-analysis.json保存五阶段逐点对比。两次均exit 0、QG pass/8 ROI；外层运行约373.960s/437.331s，均为诊断样本，不作性能结论。当前模式完整TIFF SHA为`718a903dcd0bf102fd7ea7f2be652168ad972f18bc540f5854225ee2f1bb47e3`，精确复现v22；nearest模式为`e16fd14781d86258cdeb7912abc43a3b230fa9935ec1488e0c1acda68901b605`，精确复现冻结v21。这里只建立输出等价的旧式采样反事实，不声称全部旧版内部元数据都已恢复。

24个opaque对照的跨版本不同RGB点数，在五阶段依次为 **0/0/0/17/17**：seam前、seam后、tone前均逐位一致，差异在Tone_Harmonizer内出现，finishing没有再改变。此前两个最终1色阶差异点包含在内，因此其差异已有成对tone阶段证据，不是单纯最终量化引入或首次owner取样改变。

最大差异点[6949,4372]在两版tone前RGB及owner相同，tone后RGB差为`[-0.00003904104232788086,0.000008881092071533203,0.00004756450653076172]`，finishing后不变。此点在seam后至tone前的owner会由3变2，但两版该变化一致；不能误把单次运行内部owner变化解释为跨版本误选。

本轮完成25.10提出的成对隔离与最大差异点观测，仍不把90点结果扩展为全部12,194点的逐点证明。原严格最终RGBA一致性失败保留，未新增后验容差。后续开发应转入软印章/条带全部几何有效焦点候选的同坐标原生比较，同时保持既有质量门禁；当前owner参考一致不能证明最佳焦点。

## 26. v25：软红印与横向织纹的全部有效来源比较（2026-11-03）

已新增测试专用 `focus_candidate_capture.rs`，显式配置 `RAW_EDITOR_FOCUS_CANDIDATE_REQUEST` / `RAW_EDITOR_FOCUS_CANDIDATE_DIR` 才执行。请求须匹配实际画布尺寸、world origin和scale=1；最多4窗、每窗边长不超过256。沿用完整native source映射与高质量插值，逐来源处理，导出RGB16候选、support mask和实际selected-owner mask；缺native支撑的来源显式记录，不用零填充冒充有效内容。该诊断不改变owner，不修改门槛。候选为prepared来源的tone前像素，output为tone后；没有额外gamma变换，PNG不含显式ICC，不能据此作绝对色调保真结论。

本次固定v24的ROI00/01红印窗口 `[256,128,256,256]`、`[384,128,256,256]`，以及ROI07斜笔/织纹窗口 `[9216,640,256,256]`，world origin `[-108,-49]`。9张NEF全部检查，共27个source-window记录，16个有有效支撑；来源loader额外调用9次，每个来源最多一次。该调用次数不是RAW解码数。独立release运行成功，外层619.133s，包含额外解码/导出成本，不作性能比较。QG仍pass、8 ROI；原MTF/boundary等适用性限制未取消。

离线绑定确认三个output裁片与最终TIFF逐像素一致；完整TIFF SHA仍为`718a903dcd0bf102fd7ea7f2be652168ad972f18bc540f5854225ee2f1bb47e3`，与v22相同。每个候选有效像素数与support mask核对一致；当前被选中像素均未越出对应来源支撑。两处红印各有5个有效来源，共同支撑分别45,764和45,392像素。织纹窗有6个有效来源，但六者共同支撑为0，禁止直接作六方全窗排名；DSC6481/6480/6482三张均完整覆盖，可先在这一组内比较。

已通过view_image原生查看3张output及16张候选，共19张256px图像，记录见 `output/acceptance/sleeping-9-native-v25-focus-candidates/visual-review.json`。红印当前主要来自6483，候选6484更柔和，6486/6487有更强的小纹理和边缘变化，但跨station的局部形状/位置以及tone前对比差异尚未隔离，不能宣布某张可整体替换当前owner。

织纹窗中6482占 **62,179/65,536（94.88%）**。明显横向条带在6482的映射后、tone前参考中已经存在；6481织纹较弱，6480及6479有效部分更平滑也更软。这将条带内容定位到最终tone之前，但尚不能区分原作纤维、传感器/demosaic混叠和几何重采样影响。不得仅因梯度更大就宣称更锐，也不得把较软来源直接替换为“修复”。原63孔修复、严格最终RGBA失败及完整画质未通过的历史结论保持不变。

边界/原点/溢出定向测试1项通过，真实诊断覆盖所有9来源，fmt/diff检查通过，冻结binary与全部记录源码SHA一致；未将旧版本全库结果算作此次新代码验证。精简记录在 `docs/benchmarks/sleeping-9-v25-focus-candidates.json`，详细支撑和文件SHA在 `candidate-verification.json`。

下一阶段先在共同支撑内量化局部配准偏差，并把笔画边缘、平坦噪声和周期织纹分开测量；从完整覆盖织纹窗的6481/6480/6482三张开始，确认条带属于可用细节还是周期性伪影，再决定多尺度清晰度/owner选择规则是否需要调整。诊断评分不得直接改写QG或自动选择来源。

## 27. v26：共同支撑内的配准、红边与周期纹理诊断（2026-11-03）

本阶段离线读取v25已绑定的RGB16候选，逐文件校验SHA，不重新解码RAW、不重采样图像、不改变owner/QG。代码及结果位于 `output/acceptance/sleeping-9-native-v26-candidate-metrics`。只使用红印ROI01（避免与ROI00重复计数）和织纹ROI02；来源采样仍来自此前生产RAW路径。

配准诊断在9×9 box低通后，以固定共同有效支撑、步长4、±8像素整数搜索计算NCC；候选全部位移的取样范围均先验证support mask。位移定义为candidate(x+dx,y+dy)对应baseline(x,y)，不把整数最优值当作亚像素真值。红印以6483为基准，在y96..240、x16..240比较：6484最佳(1,1)、NCC0.99367；6485(0,0)、0.99758；6486(2,1)、0.99729；6487(2,1)、0.99650。峰值与次峰差仅约1.7e-5至6.8e-5，局部纹理和边缘方向限制唯一性，不能据此直接修改几何。

红印另固定y96..224平均红色差分`R-(G+B)/2`，水平5px平滑后，按同一左边缘归一化10–90%测量：

| 来源 | 10–90%宽度（原生输出px） | 50%交点x |
| --- | ---: | ---: |
| 6483（当前主要来源） | 14.404 | 70.244 |
| 6484 | 19.199 | 71.085 |
| 6485 | 16.500 | 70.292 |
| 6486 | 14.546 | 71.689 |
| 6487 | 13.396 | 71.988 |

6483在同station三张候选中该边缘最窄，6487跨station约窄1px但伴随局部偏移。该统计包含颜料边缘本身、局部形变和色彩影响，并经过固定平滑；**不是MTF、不是全窗最佳焦点证明**。不能只凭1px差异跨station替换owner。

织纹窗仅比较完整覆盖的6481/6480/6482。中央差分方向能量及行均值周期如下；periodicity fraction是去31px趋势、Hann加窗后，在DFT bins4..64内主峰±1bin的能量比例，不是原始图像总能量占比：

| 来源 | y/x梯度能量 | 行主周期px | 峰带比例 | 行残差RMS |
| --- | ---: | ---: | ---: | ---: |
| 6481 | 1.166 | 10.24 | 67.25% | 0.009559 |
| 6480 | 0.996 | 10.24 | 39.61% | 0.003782 |
| 6482 | 2.892 | 10.24 | 92.81% | 0.030944 |

6482与6481低通全窗NCC0.99052，整数最佳(0,0)，四分块最佳位移仅一块为(0,1)，其余(0,0)。6480全窗最佳(1,0)、NCC0.95889，但四分块最佳从(-2,-2)到(0,2)，不能用单一平移消除所有差异。

主周期相位相对6482：6481差−0.07380rad，折合−0.1203px；6480差−1.84306rad，折合−3.0037px（幅度较弱，不能当成真实几何位移）。**6481和6482的同周期近同相支持纹理具有跨来源对应关系，现有证据不足以判定为伪影。** 也不能排除相干的采集/重采样混叠。直接惩罚周期能量可能损伤真实纤维；直接用强梯度作为最佳焦点也缺少对笔画、纤维和噪声的区分。fine RMS包含细节及噪声，不冒充独立噪声估计。

六项合成核验通过：恒定NCC不可测、仿射亮度NCC不变、已知(2,-1)位移符号、16px正弦周期、常量低通保持、70px logistic边缘及过渡宽度。真实来源文件SHA全部一致。此阶段没有运行新RAW验收，也没有更改生产算法或质量阈值；原MTF/boundary N/A、严格最终RGBA失败和完整画质未通过均保持。

下一步采用固定共同support做亚像素局部配准，并在多个互不重叠红笔边缘及纤维区域验证候选顺序的稳定性；先建立“边缘改善且几何无退化”的证据，再考虑owner代价的多尺度/方向性改动。周期性目前只作诊断信息，不直接扣分或抹除。输出明细：metrics.json、phase-check.json、self-tests.json。

## 28. v27：亚像素局部配准、多区域稳定性与 owner 决策（2026-11-03）

**本轮决定：不修改生产 owner 选择规则，不新增周期条带惩罚。** 对齐后候选的边缘改善不能稳定覆盖所测区域；周期纹理仍可能是来源共有的真实纤维。该决定表示证据不足以安全替换，并不证明现有owner已经全局最优。

### 28.1 固定支撑和留出验证

复用v25绑定的8张候选PNG，全部SHA重新核对；不重解码RAW、不改渲染图像。只在离线诊断中用双线性取样评价平移。红印ROI01取三个不重叠段：x16..144、y96..128/144..176/192..224；织纹ROI02取四个不重叠象限：x/y16..120、136..240。前者仍是同一笔画的三段，后者仍属同一场景，不能宣称跨画作泛化。

先分别用7×7、9×9低通，±8px整数NCC搜索，再逐级细化到1/4、1/16、1/64px。**1/64是搜索分辨率，不是实测准确度保证。** 拟合点步长4，留出点偏移(2,2)，往返分别估计；所有取样连同最大位移、插值和低通支撑的13px halo均落在有效mask内。保存原始、对齐后边缘宽度，防止忽略诊断插值影响。

运行前的protocol.json规定保守的试验筛选条件：往返误差≤0.5px，区域/尺度位移变化≤0.5px，留出NCC不下降，三个边缘段在两种低通下都改善至少5%，且搜索不碰边界。这些是本次诊断的候选筛选条件，**不是新增或放宽生产QG阈值**；即使满足也只支持后续小范围试验。实际没有替代来源同时满足。下述边缘收益反转不依赖这些筛选阈值，也足以否定按单窗均分整体替换。

### 28.2 红印分段结果

9×9低通配准后，左边缘10–90%过渡宽度改善百分比（正数更窄、负数更宽；不是MTF）：

| 候选 | 上段 | 中段 | 下段 |
| --- | ---: | ---: | ---: |
| 6484 | −30.85% | −33.81% | −29.58% |
| 6485 | −7.12% | −0.99% | −25.96% |
| 6486 | −16.90% | +12.19% | −5.71% |
| 6487 | −12.10% | +19.33% | +6.32% |

6483基准三段宽度分别15.252/14.679/14.931px。6487的最佳平移三段分别为(2.125,1.578)、(2.156,0.562)、(2.125,1.375)px，保留向量定义candidate(x+dx,y+dy)对应baseline(x,y)。左右方向约2px较一致，沿竖笔方向的位移更不稳定，符合单一方向边缘提供的约束有限；不能直接把估计差异当作真实几何误差。

另外复核同笔画右侧边缘，使用相同世界行段和对应估计平移。6487上/中段分别改善7.27%/23.84%，下段反而约增宽105.88%；6486右侧上/中段改善10.77%/25.40%，下段增宽4.91%。这里的归一化材料边缘受颜料形状、纹理、局部形变及首次阈值穿越影响，不是光学锐度标定。**中段局部优势不能外推到整窗，左右边缘也不能相互替代。**

跨区域及两种低通的位移最大差，6486为1.222px、6487为1.143px；两种低通在同区域估计的最大差分别0.625px、0.875px，最大往返误差0.797px、0.969px。更软的6484估计对低通尺度非常敏感，最大往返误差11.366px；应标作估计不可靠，不能据此给图像施加11px校正。

### 28.3 织纹区域稳定性

以6482为基准，6481在9×9低通下四块最佳位移为(−0.281,0.219)、(0.109,0.406)、(0.375,0.500)、(0.281,0.078)px。跨区域/两尺度最大位移差0.825px，最大往返误差0.759px，存在一个留出NCC轻微下降的组合（约−1.68e−5）。6480跨区域/尺度最大差7.367px、最大往返误差9.374px，弱纹理及模糊使局部估计不稳定。不能用单一亚像素平移修正全部区域后，直接将平滑来源视为更优。

这些结果与第27节同周期近同相证据一起支持：暂不把6482的强织纹当作应删除的伪影。诊断没有估计独立传感器噪声，也没有把细节残差冒充噪声或通过“更平滑”替换来源。

### 28.4 校验、产物与边界

合成测试使用已知(0.375,−0.625)连续位移核对估计方向及误差<0.05px、往返<0.1px；常量图必须不可测。上升/下降logistic边缘及分数平移验证50%交点和宽度变化<0.1px，说明本工具在合成输入上没有明显位移符号/宽度方向错误，不代表真实候选达到同等精度。

产物目录：`output/acceptance/sleeping-9-native-v27-subpixel-stability`。protocol.json保存固定方案；results.json含所有前向/反向/留出结果与原始/对齐宽度；opposite-edge.json保存右侧边缘；self-tests.json保存已知位移测试。精简验收与决策为 `docs/benchmarks/sleeping-9-v27-subpixel-stability.json`。输入身份、证据SHA及git diff检查完成；本轮仅新增离线工具与文档，未更改Rust算法，未重跑RAW或声称全库/QG新验收。

当前阶段的亚像素估计、已覆盖区域的稳定性比较和是否改owner的决策已完成。完整画质仍未通过。若继续尝试局部owner改进，应先扩大到独立笔画/边缘、提供亚像素变换的置信度和局部支撑证据，再进行有界局部试验；不能将本次中段单点收益直接转成全局焦点评分权重。

## 29. v28：按用户确认范围完成最终验收（2026-11-03）

**验收状态：`passed_for_user_confirmed_scope`。** 用户明确选择“按原性能目标和既有质量门禁验收”，本节完成同组连续9 NEF、scale=1原生TIFF的性能、资源、来源证据、既有Quality Gate及有界原生视觉验收。它不要求所有画质指标都可测或清除全部素材相关degraded，也不声明全部45张不连通素材可合成、画面处处最佳对焦或其他数据集已通过真实渲染。

本次未继续修改owner选择算法。第25节起额外设定的“补孔后24正控最终RGBA必须与含孔旧版逐位相同”诊断仍保持failed，2点1色阶变化及第25.11节的tone阶段成对归因均保留。该额外诊断不属于此次用户确认的原性能/既有QG门禁；没有回写旧结果、伪造旧版像素一致或给QG增加后验容差。第26–28节的候选研究与完整最佳焦点未证明状态同样保留。

### 29.1 当前版本及正式检查

复用并核验当前完整源码对应的v25 release，冻结为 `/private/tmp/raw-editor-v28-final-acceptance-release-binary`，SHA **`e50f6a4c9f77b49ed20e6a20a36976a2fa4b3e5259d0598cd55aa729ab29a72c`**。正式运行前后核对binary、全部记录Rust源码、实际9张NEF字节；生产cargo check、fmt、diff检查通过。当前冻结版本完整library回归 **1017 passed / 0 failed / 40 ignored，112.84s**，不是沿用v22的1016项结果。数据集契约执行 `--require-data --hash-inputs`，清单内 **11个数据集全部通过**；这只证明数据契约与哈希，不等于它们全部完成渲染验收。

cold/hot串行独立进程，清理全部继承RAW_EDITOR诊断配置，point trace、source visual、candidate capture与nearest-only开关全部关闭。预算保持source cache **2,400,000,000 B**、native fusion **4,294,967,296 B**、RSS硬门 **25,769,803,776 B**。这是明确的验收profile，不代表产品默认512MiB source cache下具有相同命中率或耗时。未purge OS page cache；cold/hot是运行次序标签，热跑并不承诺一定更快。

### 29.2 同环境基线与性能结果

本次当前版耗时比历史记录高，故追加同会话、串行的冻结v19优化前基线，不把旧时间直接当成当前环境对照。控制运行binary SHA `e4dce3a0ef48313a7bf1bd4deb2301119e61cf91ca54c1fc492bb239f6681c9a`，仍为相同9输入、配准/closure/density和QG策略，成功复现旧TIFF `e16fd14781d86258cdeb7912abc43a3b230fa9935ec1488e0c1acda68901b605`。基线保留旧孔，当前版具有已单独验证的覆盖修复和tone变化；不是跨版本像素相同的性能实验。

| 指标 | 同环境v19控制 | 当前cold | 当前hot |
| --- | ---: | ---: | ---: |
| Rust子进程wall（s） | 652.684 | **431.113** | **449.322** |
| 外层execution wall（s） | 653.501 | 431.801 | 449.503 |
| 完整RAW解码次数 | 85 | **67** | **67** |
| native fusion次数 | 6 | 6 | 6 |
| native fusion wrapper累计（s） | 302.182 | 243.621 | 248.182 |
| OS maximum RSS（B） | 9,960,488,960 | 9,785,131,008 | 10,348,183,552 |
| Rust user CPU（s） | 2530.195 | 2208.363 | 2153.317 |

相对此次顺序控制运行，当前cold/hot的实测wall下降 **33.95% / 31.16%**，完整RAW解码下降 **21.18%**。RSS分别−1.76%/+3.89%，因此**不声明峰值内存稳定下降**；三次均在原RSS门限内。这是一次候选cold/hot加后续一个控制的观察，不是随机顺序重复实验或统计置信区间，不能推广为所有素材的固定提速比例。也不把wrapper累计或历史“Focus ownership solved”时间误称为纯图割内核耗时。

当前两次stat/hash/EXIF均9次，source cache峰值2,211,124,992 B，native cache峰值2,948,607,844 B，均受原预算约束。source cache QG hit/miss为51/17，native cache hit/miss为15/6；raster-only payload skips=6、owner payload builds=12、owner source loads=36。ROI cache本次0 hits/32 misses，不能宣称此冷/热跨进程命中了运行内ROI缓存。

### 29.3 输出、质量与来源证据

cold/hot去除计时、运行标识及RSS观测后业务报告完全一致；两张TIFF完整SHA都为 **`718a903dcd0bf102fd7ea7f2be652168ad972f18bc540f5854225ee2f1bb47e3`**。尺寸 **15897×6720**，16bit RGBA、sRGB ICC、ExtraSamples[2]，artifact validation pass；完整文件hash与执行/报告SHA及读取前后stat绑定。

两次既有QG均 **pass / 8 ROI**，与v22的实测QG结果及策略一致，技术不可测与owner反查失败没有新增。MTF0/512和boundary0/372仍是内容不适用，grouping/low-sharpness的degraded继续呈现，不伪装为可测或全清晰。闭环、density、几何来源和资源策略核对通过。

最终TIFF与已绑定的v22/v25输出逐字节相同，因此复用对应的实际来源证据：9张RAW均有正贡献、covered owner-zero=0、covered pixels=97,858,771；63个已知孔actual render source→最终QG owner→TIFF链成立，旧有效覆盖未丢失，新增覆盖553。不是从颜色或全图census倒推逐点owner。原始绑定、24个变化原生窗口及8对output/reference的真实review SHA均重新核对，历史检查没有改写为新运行。

本轮额外从最终hot TIFF生成并实际查看 **2400×1014** overview（工具显示缩至2048×865），未见新增明显宽幅矩形接缝、断栏或重影；外覆盖透明台阶保留。overview只用于整体结构，不代替1:1检查；原生细节沿用同一输出SHA的24窗、8对来源及后续候选研究。9张子集只覆盖原作下部，不是全幅原作验收。

### 29.4 归档与收口

完整机器可核验结果：[final-acceptance.json](/Users/dp/Desktop/raw-editor/output/acceptance/sleeping-9-native-v28-final-acceptance/final-acceptance.json)。精简指标与边界：[sleeping-9-v28-final-acceptance.json](benchmarks/sleeping-9-v28-final-acceptance.json)。冷/热执行、报告、hash、全库日志、数据契约和overview均归档在同一v28目录，同环境控制单独放在 `sleeping-9-native-v28-baseline-control`。

按本次确认范围，验收已完成，可以结束这一轮性能优化验证。后续跨数据集性能、默认小缓存profile专项、更多独立笔画的最佳焦点与P2进一步优化属于独立后续任务，不冒充本次已完成项。

## 30. 已验收基线固定与工作区集成（2026-11-03）

第 29 节的用户确认范围、性能数据和 Quality Gate 结论保持不变。当前 497 个 Rust 源码文件与
冻结构建逐个 SHA-256 一致；9 个输入、2 个 TIFF 和来源/视觉记录共 15 项重新核对通过。
冻结二进制已复制到工作区的本地归档，避免仅依赖 `/private/tmp`。

本轮对现有未提交内容先保存 patch、清单、哈希和原文件，再按 keep/adapt/test-only/exclude
登记全部 237 个展开文件。其他前端、亮度工具、设置及原始诊断保留在工作区，不纳入堆栈基线。
已同步唯一执行任务清单，其他数据集的待验收项保持未完成。

当前 Rust 全量回归为 1017 passed / 0 failed / 40 ignored；前端类型、lint、构建和 Node
契约通过。集成发现报告 schema 的 Node 契约遗漏 `performance`，以及亮度工具 11 个新文案键
未进入语言包，已分别同步契约/设计文档和 12 个语言包；生产判据、owner 规则和 Rust 源码未修改。
全库格式检查存在既有 48 项告警，未通过全库格式检查；不以批量格式化改写其他工作。

基线身份、逐文件归属、命令结果和后续边界统一见
[已验收基线与集成记录](focus-stack-accepted-baseline.md)。本次集成通过不扩展为全部素材、
所有指标可测、真实接缝识别或跨平台正式发布通过。

独立本地基线分支为 `codex/accepted-stack-v28-baseline`，提交 `504d904f`；当前 `v1` 和暂存区
保持不变，未推送。macOS app 已通过本机离线构建，并实际完成独立 NEF 副本导入、曝光调整和
8256×5504 sRGB JPEG 导出。源文件未改动；这项单图桌面冒烟不代替其他素材或平台的发布验收。

固定 Rust 1.96.1 的完整库测试另行复跑为 1017 passed / 0 failed / 40 ignored（114.77s），
与根目录工具链的检查均通过。最终集成摘要见 `docs/benchmarks/workspace-integration-2026-11-03.json`。
打包后的并发接缝更新只补充 Node/类型/lint/Web 构建验证；app 冒烟仍绑定实际受测二进制，不混用版本证据。

## 31. 默认缓存预算与跨素材验收（2026-11-03）

本阶段使用已固定的 v28 release 测试二进制，Rust 源码身份保持不变。清除继承的 RAW_EDITOR 配置，
不设置 source/native fusion 缓存覆盖值：真实默认分别为 **512MiB / 1GiB**，并非第 29 节的
2.4GB / 4GiB 验收 profile。质量判据、owner 规则、alpha/ICC 与 RSS 门限保持不变。

同一 9 张 NEF 先执行串行独立冷/热进程，均为 scale=1 和原生 TIFF；不 purge 系统页缓存，
因此冷/热只表示运行顺序。报告将分别列出质量与预算验证、性能观察，不能把默认预算与旧大缓存
profile 的差异全部归因到单一缓存，也不要求热跑必然更快。

本轮日志与来源绑定写入 `output/acceptance/sleeping-9-native-v29-default-cache/`。
开始时已保存当前 status、tracked patch 和 246 个展开文件的哈希，现有未提交内容保持原状。
完整帧可能大于默认 source 预算，须检查 oversized bypass 与实际 decode/fusion 计数后再作性能结论。

同组默认配置验证结束后，再按固定 manifest 对 Langyuan-10、Wenyuan-10 与兰亭 45 张执行
真实入口验证。跨素材记录写入 `output/acceptance/real-materials-v30-default-cache/`。
这些是不同路径的验证：原生 layered 门禁拒绝继续算失败；legacy 行为成功不能替代 Quality Gate 通过。

### 31.1 默认配置冷跑与问题定位

冷跑质量与预算验证通过：Quality Gate `pass / 8 ROI`，门禁策略和逐项质量结果与 v28 一致，
TIFF SHA 仍为 `718a903dcd0bf102fd7ea7f2be652168ad972f18bc540f5854225ee2f1bb47e3`。
完整运行 1724.400s，完整 RAW 解码 176 次、native fusion 21 次（wrapper 累计 1366.420s），
OS maximum RSS 8,449,048,576 B（约 7.87GiB）。stat/hash/EXIF 各 9 次，SourcePreflight 复用仍有效。

source cache：预算 536,870,912 B，0 hits / 167 misses / 167 oversized bypasses，峰值驻留 0 B。
校正后的 8312×5542 RGB32F 帧为 552,781,248 B（527.17MiB），超过默认预算约 15.17MiB。
native fusion cache：预算 1,073,741,824 B，0 hits / 21 misses / 19 evictions，峰值
1,029,928,412 B，最多驻留 1 项。该访问顺序下未形成有效复用。

与第 29 节大缓存 cold 的历史观察相比，本次耗时约 4 倍、RAW 解码 67→176、fusion 6→21；
RSS 约下降 13.65%。两项预算同时变化、未做随机顺序重复实验，不能把全部差异只归因于 source
预算，也不把该比例推广到其他素材。**质量和预算通过不等于默认配置的性能优化已经完成。**

冷跑业务 harness 正常通过，但最初的后处理校验把访问时间也纳入 `stat` 元组比较；读取 TIFF
自身改变访问时间，导致观测脚本报错。已保留原脚本与失败日志，改用 dev/ino/size/mtime_ns/ctime_ns
绑定，离线重读 TIFF 完成校验，未重跑 RAW、未改写业务报告、未放宽质量门禁；热跑使用修正后的
观测脚本。差异记录见本地 `validation-adjustment.json`。

### 31.2 默认配置冷/热收口

[默认配置验收摘要](benchmarks/sleeping-9-v29-default-cache-acceptance.json)记录两次完整运行。
两次原生 TIFF 均与 v28 逐字节相同，剔除运行标识、计时和 RSS 采样后的冷/热业务报告完全一致。
input、grouping、intra-station、fusion、topology、station relations、closure、residual warp、tone、
composition、Quality Gate 与 v28 的非计时证据逐项一致。原有 owner/source 与原生 1:1 视觉记录
继续绑定相同输出 SHA；本轮计时运行未开启额外逐点 capture，不伪称新采集了一轮逐点证据。

| 指标 | v28 大缓存 cold / hot（历史观察） | 默认 cold | 默认 hot |
| --- | --- | ---: | ---: |
| source / native fusion 预算 | 2.4GB / 4GiB | 512MiB / 1GiB | 512MiB / 1GiB |
| native wall（s） | 431.113 / 449.322 | 1724.400 | 1262.314 |
| 完整 RAW 解码 | 67 / 67 | 176 | 176 |
| native fusion 次数 | 6 / 6 | 21 | 21 |
| source cache hit / miss | 64 / 58（两次相同） | 0 / 167 | 0 / 167 |
| native cache hit / miss | 15 / 6（两次相同） | 0 / 21 | 0 / 21 |
| OS maximum RSS（B） | 9,785,131,008 / 10,348,183,552 | 8,449,048,576 | 8,736,735,232 |
| user CPU（s） | 2208.363 / 2153.317 | 6559.318 | 6525.557 |
| Quality Gate / ROI | pass / 8 | pass / 8 | pass / 8 |

默认 hot 的 wall 比 cold 小，但两次工作量计数及 user CPU 接近，缓存均 0 hits，不能解释为
跨进程命中或消除了重复计算。所有缓存峰值与原 RSS 门限均满足，network requests 为 0。
既有内容不适用和 degraded 状态不变。**本阶段结论是质量和预算验收通过；默认预算下的性能目标仍需优化。**

下一步缓存开发应针对观测到的两个问题，保持 512MiB / 1GiB 预算与实际驻留字节计账：

1. 对完整帧超过 source 预算的情况，评估分块或有容量上限的无损后备缓存；避免用量化、缩放或降低 RAW
   解码质量来换取表面命中率。保留 SourcePreflight 的 stat/hash/EXIF 各一次。
2. 对 native fusion 跨机位循环访问造成的淘汰，比较访问顺序收敛与有容量上限的无损结果复用；完整保留
   owner、Coverage、source/geometry 状态和失效条件。以减少实测 21 次重复融合为目标。
3. 在同一组输入上重新验证 cold/hot、TIFF SHA、全部现有质量证据、磁盘/内存预算及实际解码/fusion 计数；
   再扩展其他素材，不能仅以缓存实现或单元测试通过勾选性能完成。

v28 与默认配置并非整个报告完全相同：默认配置的 `max_resident_virtual_tiles` 为 2（v28 为 4），
重复融合使 `fusion_low_sharpness_region` 日志条目由 6 增至 21；grouping motion 条目仍为 4。
去重后的降级证据完全一致，整体结果仍为 degraded。原始 21 条记录全部保留，没有为了比较而删改报告。
两次默认配置之间的完整非计时业务报告则相同。


### 31.3 跨素材原生入口验证

沿用同一冻结二进制、真实默认缓存和生产解码入口，依次运行固定 manifest 的 Langyuan-10、
Wenyuan-10 和兰亭 45 张，全部 scale=1。每套记录输入 SHA、命令、环境、二进制、报告和进程
资源数据，运行前后复核输入、参考图及 497 个 Rust 文件；不更新既有 regression baseline。
各素材只执行一次，不称为它们的冷/热性能验收。

阆苑女仙 10 张被几何门禁拒绝：分为 4/3/3 帧的 3 个机位，接受跨机位关系 0 条，仍有 3 个
不连通分量。3 对候选均停在 `polished_fit_failed`，双向匹配数分别为 6、14、2，缺少可拟合的
分布式支持；诊断另记录 3 次尺度越界及光度、边缘强度、方向不一致。没有生成 TIFF，closure
与 Quality Gate 均为 `not_run`。不得将拒绝解释为已完成画质验收或素材不可用。

文苑图 10 张同样被几何门禁拒绝：3 个机位中 0→1 关系通过，452 个双向匹配覆盖 32 个网格；
0→2、1→2 两对最终候选只有 2/1 个双向匹配、0 个有效网格，均未完成拟合，报告同时保留
2 次尺度越界诊断。最终 2 个分量，未生成 TIFF，closure 与 Quality Gate 为 `not_run`。
已接受关系保留为后续正对照，不能为连接剩余机位而放宽尺度或证据门槛。

N4 的后续步骤是先核对 source/native/world 坐标、预测重叠及初始模型，再按失败机位对检查
多区域双向支持；增加独立留出区域和负对照后才改变生产行为。候选提议与 coarse 匹配不计作
生产关系证据。保持当前 9 张 NEF 正对照、owner/source、Coverage、closure 和 Quality Gate 不变。
阆苑女仙 84 张、文苑图 126 张的全量扩展继续等待子集几何与画质通过。

复现入口为本地归档的 `sleeping-9-native-v29-default-cache/run-default.py`（cold/hot）和
`real-materials-v30-default-cache/run-materials.py`（三套素材串行），每次具体命令见对应
`execution.json`。这些脚本使用固定归档目录并拒绝覆盖；再次实验应使用新的归档目录和标签。
原始 Stack_Report、非零退出及 harness 失败日志均保留，汇总脚本成功只表示证据整理完成。


兰亭 45 张 JPEG 完成 legacy source-scan 渲染：排序 45/45、流式处理 45/45，harness 的覆盖区域
ownership 检查成立，生成 33821×13852、16-bit sRGB TIFF。原生子进程退出 0，但外层验收退出 1：
报告 `alpha_preserved=false`、`ExtraSamples=[]`，未满足本轮明确开启的 unassociated coverage alpha
合同（要求 `[2]`）。通用 TIFF `artifact_validation=pass` 只表示其已有格式检查通过，不能覆盖
这一失败。Quality Gate 仍为 `insufficient_evidence / 0 ROI`，因此行为保护验收和画质验收均不能
勾选通过；没有关闭 alpha 校验或修改 baseline 来消除失败。

兰亭的参考图指标 median sharpness ratio 为 1.11067、P10 为 0.91593，仅是已有比较指标，不能
替代 Quality Gate。人工查看 overview 与 top-center、bottom-left、bottom-right 三张缩略区域，
可见画作、外框和背景，以及外沿阶梯状黑边、部分竖向色调与框线/背景不连续。该项只记为视觉
问题观察，不声称原生 1:1 或无瑕疵验收通过。图像 SHA 与检查范围见本地 `lanting/visual-review.json`。
后续 N6/N7 需要补齐 legacy Coverage/alpha 导出合同及相应来源、画质证据，保留当前失败作回归。

| 素材 | 原生 wall（s） | OS maximum RSS（B） | 原生子进程 / 外层验收退出码 | 结果 |
| --- | ---: | ---: | --- | --- |
| 阆苑女仙 10 NEF | 191.795 | 10,568,269,824 | 101 / 1 | 3 分量断连，未导出，QG not_run |
| 文苑图 10 NEF | 212.492 | 9,267,986,432 | 101 / 1 | 2 分量断连，未导出，QG not_run |
| 兰亭 45 JPEG | 341.945 | 9,887,711,232 | 0 / 1 | 已导出，alpha 合同失败，QG insufficient_evidence |

三套实际运行和失败归档已完成，**跨素材通过数量为 0/3**。精简报告见
[跨素材默认配置验收摘要](benchmarks/real-materials-v30-default-cache-acceptance.json)。
阆苑/文苑各 20 次 RAW 解码；报告中的 native fusion cache 计数 0 不代表准备阶段没有融合。
兰亭是 JPEG 输入，RAW 解码计数为 0；source cache 45 misses / 45 oversized bypasses，
不能把这个 0 当成无需加载图像。三次峰值 RSS 均低于原 24GiB 门限，所有原始输入与参考文件保持不变。

本阶段未改生产代码或门禁，未提交或推送；任务清单仅勾选运行和证据归档完成，默认缓存有效复用、
跨素材几何、legacy 输出合同及画质泛化仍为待办。当前工作区并发产生的亮度、接缝和 package 修改
保留；497 个 Rust 文件与固定二进制身份再次核对，文档/JSON 校验与 `git diff --check` 通过。


## 32. 默认预算下的无损后备复用（2026-11-03，9 张子集验收完成）

本轮保持 source 512MiB / native fusion 1GiB 的 RAM 预算，不改变像素类型、解码质量、owner
选择或 Quality Gate 判据。对占单级预算超过四分之一的大对象使用运行内无损后备存储，小对象
继续使用内存 LRU；来源缓存另限 8GiB、融合缓存另限 4GiB 临时磁盘容量。新增磁盘容量是明确的
资源成本，不能表述为总资源只占 512MiB / 1GiB。活动解码/读取结果和 OS 页缓存仍由全进程 RSS
验收约束，不纳入缓存驻留字节冒充零成本。

后备文件使用匿名临时文件，先淘汰到容量可容纳本次写入，再写入原始 f32/u16/u8 位模式，不量化、
缩放或重新融合。每次读取验证长度和 SHA-256；文件失败视为缓存未命中，重新走原生产计算路径。
clear、错误退出或对象释放关闭句柄；文件没有可供后续运行查找的路径，不引入跨运行缓存命中。

来源键仍绑定 SHA、尺寸和处理指纹。融合结果保留完整 Coverage、ownership legend、confidence、
textured、sharpness evidence、source sampling/refinement、displacement、fusion 记录和 sampling origin；
几何键与融合前输入状态仍参与失效判定。后备元数据、来源采样快照和相关容量计入原 RAM 预算。
磁盘 `max/resident/peak_bytes`、hits/writes、读写字节与耗时、failures/evictions 单独写入
`performance.*_cache.backing`，旧报告缺少该字段时继续使用 serde 默认值。

验证顺序：定向缓存/证据/损坏与容量测试 → 全库 Rust 回归与数据/质量契约 → 冻结新二进制和源码
SHA → 同组 9 张 scale=1 冷/热运行 → 与 v29 默认预算及 v28 相同 TIFF/来源证据比较。本轮归档为
`output/acceptance/sleeping-9-native-v31-bounded-backing/`，开始时保护了 255 个展开的现有修改文件。
本节记录同组 9 张子集的完整冷/热验收，不扩展为其他素材已通过。


`peak_entries` 表示缓存条目数，后备条目只持有元数据，不等于驻留完整 raster；新增
`native_fusion_cache.peak_memory_entries` 单独记录完整内存 payload 的峰值数，
`max_resident_virtual_tiles` 使用后者再加活动 tile，避免把磁盘条目误报成驻留图像。

磁盘容量按后备文件 payload 字节计账，文件系统块/句柄开销不计入该 payload 指标；
读写时间包括完整性摘要处理。SourcePreflight 的原始文件 hash 仍只计每源一次，后备内容的
SHA 校验属于独立缓存 IO 成本，不混为“完全没有重复哈希”。


### 32.1 冷跑与实现验证

最终缓存定向测试 50 passed / 0 failed / 2 ignored，全库 release 测试 1021 passed / 0 failed /
40 ignored；fmt、生产 cargo check、数据集/质量契约及 8 项回归拒绝规则通过。生产 check 保留
188 条编译告警（含测试用 source cache 构造器在非测试构建下未使用），不宣称零告警。

冷跑 native wall 418.756s（v29 默认 cold 为 1724.400s），RAW 解码 176→18、native fusion 21→6。
source cache 113 hits / 9 misses，其中 Quality Gate 68 hits / 0 misses；native cache 15 hits / 6 misses。
3 次输入状态失效仍触发重算，未绕过第一次 native registration 后帧状态变化的保护。
SourcePreflight stat/hash/EXIF 各 9 次，后备 IO failures/evictions 均为 0。

QG pass / 8 ROI，station owner evidence 8；TIFF SHA 与 v28/v29 完全一致。
input、grouping、intra-station、fusion、topology、station relations、closure、residual warp、tone、
composition、Quality Gate 的非计时证据逐项一致。low-sharpness 降级日志由重复融合的 21 条回到
6 条，grouping motion 仍为 4 条，原有 degraded 与内容不适用状态保留。

缓存 RAM 峰值：source 10,152 B、native 46,547,620 B；后备文件 payload 峰值分别
4,975,031,232 B（4.633GiB）和 2,902,063,320 B（2.703GiB），均低于独立 8GiB / 4GiB 上限。
本轮 source/native 读取分别 62,464,281,024 / 14,510,316,600 B，说明消除 RAW 重解码有实际 IO
成本；对应读耗时 28.820 / 7.466s，写耗时 2.334 / 2.660s。这些计时不能与包含它们的阶段时间
相加冒充总时间。

OS maximum RSS 8,745,779,200 B（8.15GiB），比 v29 cold 高约 3.51%，仍在原 24GiB 门限内。
因此结论是减少重复计算并降低总耗时，**不是全进程峰值内存必然下降**。冷/热最终结论见下节。


### 32.2 冷/热最终验收与资源成本

| 指标 | v29 默认 cold / hot | v31 默认 cold / hot |
| --- | --- | --- |
| native wall（s） | 1724.400 / 1262.314 | **418.756 / 451.682** |
| 相对 v29 同阶段耗时变化 | — | −75.72% / −64.22% |
| RAW 解码次数 | 176 / 176 | **18 / 18** |
| native fusion 次数 | 21 / 21 | **6 / 6** |
| source cache hit / miss（两次相同） | 0 / 167 | 113 / 9 |
| native cache hit / miss（两次相同） | 0 / 21 | 15 / 6 |
| fusion wrapper（s） | 1366.420 / 997.240 | 255.175 / 277.686 |
| user CPU（s） | 6559.318 / 6525.557 | 1862.479 / 1850.217 |
| OS maximum RSS（B） | 8,449,048,576 / 8,736,735,232 | 8,745,779,200 / 9,497,903,104 |
| Quality Gate / ROI / station owner evidence | pass / 8 / 8 | pass / 8 / 8 |

热跑两层命中/读写次数、缓存 RAM/payload 峰值均与冷跑相同，IO failures/evictions 均为 0。
热跑 source/native 读耗时 29.842 / 8.075s，写耗时 2.187 / 2.866s；每次累计读取约 71.69GiB、
写入约 10.04GiB 后备 payload。这是磁盘 IO 与减少 RAW 解码/重复融合之间的实测交换，不是零成本缓存。
热跑 RSS 比 v29 hot 高 8.71%，两次仍低于未改变的 24GiB 门限。没有把缓存 RAM 降低写成进程 RSS 降低。

冷/热 TIFF 均为 15897×6720 RGBA16 sRGB，SHA 仍为
`718a903dcd0bf102fd7ea7f2be652168ad972f18bc540f5854225ee2f1bb47e3`。
剔除运行标识、计时、RSS 采样后的完整冷/热业务报告相同；第 32.1 节列出的固定证据两次均与
v29 相同，去重后的降级证据也相同。保留原始 6 条 low-sharpness 和 4 条 grouping motion 日志，
不抹除 degraded。本轮没有重新运行全像素 owner 普查或逐点视觉 capture；原有 source binding
与原生 1:1 评审继续绑定相同输出，当前 owner/source、geometry 和 QG 证据另由实际报告核对。

本次是独立进程顺序运行的一组 cold/hot，没有 purge OS 页缓存，匿名后备文件不跨运行复用。
hot 比 cold 慢约 33 秒，但工作量计数一致、user CPU 接近，不能用“热跑必然更快”解释验收。
与 v29 的时间百分比为单次历史对照观察，不推广为其他素材或硬件的保证。

新冻结测试二进制 SHA 为 `88ce505d644b203eb7283cd0a81f83b1b268f0e5f51cf387a264444fb1f04abc`，
固定 Rust 1.96.1；源码清单为 499 个文件（相对 v28 为 6 个既有文件修改、2 个新增文件）。
完整结果见 [v31 默认预算验收摘要](benchmarks/sleeping-9-v31-bounded-backing-acceptance.json)，
本地 `build.json`、`source-final-intent.json`、cold/hot 的 execution/analysis/Stack_Report 和测试日志
均保留。最初 3 个按纯像素设置预算的单元测试失败记录也保留；最终测试已按包含元数据的实际驻留成本验证。

**本轮同组默认预算复用与冷/热验收完成。** 既有 v28 独立 Git 基线未改写，当前修改未提交或推送。
下一阶段仍按任务清单推进跨素材几何与 legacy 输出合同；v30 的失败记录保留，未声称 v31 已通过
那些素材。扩大素材时须同时比较后备写入、命中和耗时，不能仅凭这组 NEF 的收益推广所有路径。

## 33. 跨素材机位断连：局部坐标修复与候选证据保留（2026-11-04，v32）

本轮从冻结的 v31 二进制重新运行阆苑女仙、文苑图各 10 张 NEF（scale=1、默认缓存预算），
同时保存原生灰度与 Coverage 平面。原基线仍分别为 3 / 2 个连通分量，接受关系 0 / 1，
Quality Gate 和 closure 均 `not_run`，未写出 TIFF。没有把真实素材存在重叠等同于已有合规关系。
基线原生子进程耗时 238.507s / 267.571s，RSS 9,958,670,336 / 9,212,608,512 B；本轮启用
诊断 PNG 保存，不能直接拿这些时间与未保存平面的历史运行比较。

### 33.1 已定位并修复的坐标混用

Layered 路径通过 `focus_anchor_render_frames` 将每台机位的 source pose 转换到该机位的 anchor
坐标；`geometries[station].tile_to_world` 在这里表示该局部坐标内的裁切平移。
`tile_homographies` 则还包含 `anchor_to_world`，用来表达跨机位布局。此前 source match seed
错误地把后一种位姿与前一种 source pose 复合，重复带入尚待验证的全局布局，造成有真实 source
匹配时仍出现 `no_predicted_overlap` 或极少原生支持。

修复后的 source→tile 复合为：

`H_tile = C_right⁻¹ × L_right × (S × H_raw × S⁻¹) × L_left⁻¹ × C_left`

其中 `C` 是对应机位的局部 tile 裁切矩阵，`L` 是 source→station-local，`S` 是渲染比例。
函数改为接收 `FocusVirtualTileGeometry`，调用方明确传入 `geometries[left/right]`，不再传全局
`tile_homographies`。正常 source RAW 关系仍只是搜索起点；sequence bridge、无 consensus 的
coarse bridge 和不可逆/无效输入仍拒绝，最终必须重新测量原生像素并通过原共享门禁。

没有调整 inliers≥24、median≤3px、scale∈[0.95,1.05]、hull≥0.20、光度/边缘/方向/凸性、
closure 或 Quality Gate 阈值；没有修改 owner 选择、融合、RAW 解码和默认缓存预算。
也没有以拟合出的倍率重定义坐标单位来使 scale 门禁通过。

### 33.2 候选生命周期与报告修复

原报告只留下每对机位的最后一次尝试。实际已有数百个双向支持、但被 scale 拒绝的模型，会被
之后的 `global_coverage_similarity / polished_fit_failed` 覆盖，甚至被汇总成“未测量”。
现在 `candidates[].previous_attempts` 按执行顺序保存扁平历史，含各次 seed、原生支持、模型、
共享门禁结果、canonical 方向和既有 holdout/Local_Scale 证据。搜索预算不变，历史不参与关系
选择或连接图。单次尝试省略该字段，旧报告仍可读取。

日志分开记录 `model_fitted` 与 `gate_passed`。`unmeasured_count` 只统计所有尝试均无拟合的机位对；
若最后一次失败但较早已有拟合，则汇总为 `measured_relation_not_accepted`，具体门禁原因保留在
hypotheses 和 `rejected_by_reason` 中。它是观测修正，不是新增接受关系。

新增定向测试覆盖局部裁切与全局位姿混用、正反 source 方向、scale=1/0.5、无效/无实测桥接，
以及多次尝试历史的顺序、无嵌套、JSON 往返和未测量计数；原有 source seed 测试适配明确的
geometry 参数。首次编译暴露的旧测试签名错误日志保留，最终验证和真实复验结果在下节记录。

### 33.3 真实跨素材复验：搜索起点已修复，几何门禁尚未通过

| 子集/机位对 | source 尝试拟合成功（前→后） | 修复后最大拟合支持 | 选中假设的 canonical scale 范围 | 独立 holdout median error（px） |
| --- | --- | --- | --- | --- |
| 阆苑 0→1 | 0/4 → 4/4 | 295 | 0.908015–0.913465 | 0.887–1.011 |
| 阆苑 1→2 | 0/4 → 4/4 | 542 | 0.912719–0.913939 | 1.177–1.228 |
| 文苑 0→2 | 0/4 → 4/4 | 624 | 1.114653–1.119005 | 0.681–0.840 |
| 文苑 1→2 | 0/4 → 4/4 | 552 | 0.889338–0.892828 | 1.006–1.276 |

“最大拟合支持”来自 matcher 的 `fitted_inliers`；后两列来自共享 selector 选中的假设，
不是把不同模型的计数混成同一条关系。canonical 方向由内容确定，因此两对文苑的倍率一大一小，
不能直接用表中的机位编号顺序解释倍率方向。每个选中 source 假设均仅因
`station_relation_scale_out_of_range` 被拒绝，未把它们计入 accepted。

| 指标 | 阆苑 v31 → v32 | 文苑 v31 → v32 |
| --- | --- | --- |
| 连通分量 | 3 → 3 | 2 → 2 |
| accepted 关系 | 0 → 0 | 1 → 1（原关系记录完全相同） |
| unmeasured 机位对 | 3 → 0 | 2 → 0 |
| Quality Gate / ROI | not_run / 0 | not_run / 0 |
| closure / TIFF | not_run / 未写出 | not_run / 未写出 |
| native wall（s） | 238.507 → 276.656 | 267.571 → 319.794 |
| OS maximum RSS（B） | 9,958,670,336 → 9,544,400,896 | 9,212,608,512 → 8,355,823,616 |

两套输入均逐张校验 RAW 与参考成片 SHA 不变；每套三个机位的原生灰度、质量平面、Coverage
共 9 个 PNG，在修复前后全部逐字节相同。时间增加约 38.15 / 52.22s，来自恢复后的原生搜索工作；
本轮没有宣称跨素材加速。RSS 为单次观测，不把差值推广为稳定节省。
`unmeasured` 下降是修正历史覆盖后的观测结果，**不是图连通性改善**。

原生样本和独立 checkerboard holdout 拟合均支持越界的倍率。例如阆苑 0→1 的 source 候选样本
Local_Scale 为约 0.904–0.918，holdout 模型在同批采样位置约 0.904–0.918；1→2 也均低于 0.95。
阆苑非相邻 0→2 的先验还存在光度、边缘强度与方向拒绝，未抹去这些原因。
只读 source-frame 审计显示：两套输入分别全部为 74mm / 70mm 等效焦距，均为 8258×5507 的
校正后 source 网格；已有 source 模型投到 anchor 坐标后的倍率也在约 1.09–1.12。
这些网格评估点仅诊断模型倍率，**不是新增实测对应点，也不参与生产连接**。

因此本轮结论是：明确修复局部/全局坐标混用及证据覆盖问题，恢复 16/16 次 source 拟合；
真实子集仍未达到 N4 的连通与闭环验收。下一步应核对原生采样倍率、anchor 构建与尺度门禁的
物理量契约。如需引入独立标定后的共同采样网格，须单独定义和验证 RAW fidelity、重采样、
source/native/world 变换及 holdout，不能利用被拒模型的倍率把 scale 除回 1，不能把残差尺度
偷偷替换为原门禁，也不应继续盲目扩展 top-K、强制 bridge 或升级到 84/126 张运行。

本地证据：`output/acceptance/cross-station-v32/geometry-comparison.json`、
`source-frame-audit.json`、两套 baseline/candidate execution、Stack_Report、原生平面和日志。
外层真实验收均退出 1、Rust child 均退出 101；驱动脚本的 0 只表示两次拒绝均已归档。

### 33.4 集成测试与 9 张 NEF 基线保护

新增定向测试 3 passed / 0 failed；完整 release 库测试 1024 passed / 40 ignored / 0 failed。
生产 `cargo check --locked --offline --lib`、格式检查及 quality/datasets/regression 三套契约均通过。
冻结二进制 SHA 为 `36a5cc87887871ded7b75ee50b1afdfc002931a231e671e1372c28aa35c8dd68`，
固定 Rust 1.96.1，500 个 Rust 文件。共享门禁、selector、owner/fusion 和缓存实现未变。

同组 9 张 NEF 已完成默认预算下的独立进程冷/热回归，OS 页缓存未清除：

| 指标 | cold | hot |
| --- | --- | --- |
| native wall（s） | 395.372 | 392.813 |
| OS maximum RSS（B） | 9,497,214,976 | 8,276,754,432 |
| RAW 解码 / native fusion | 18 / 6 | 18 / 6 |
| source / native cache hits | 113 / 15 | 113 / 15 |
| Quality Gate / ROI | pass / 8 | pass / 8 |

两次 TIFF 均为 15897×6720 RGBA16 sRGB，SHA 继续为
`718a903dcd0bf102fd7ea7f2be652168ad972f18bc540f5854225ee2f1bb47e3`。
input、grouping、topology、station relations、closure、intra-station、fusion、residual warp、tone、
composition、Quality Gate、virtual tiles 的非计时内容分别与 v31 同阶段报告完全相同，覆盖原有
owner/source 证据；冷/热完整非计时业务报告也相同。默认 RAM 512MiB / 1GiB、后备 8GiB / 4GiB
均未超限，IO failure 为 0，RSS 通过原 24GiB 门限。单次时间观测不构成进一步性能提升承诺。
没有新做逐点视觉 capture；既有原生 1:1 评审继续绑定相同 TIFF SHA，不扩大其验收范围。

结果见 [v32 复验摘要](benchmarks/cross-station-v32-local-source-seed-acceptance.json)，详细资料在
`output/acceptance/cross-station-v32/`。本轮开始前 259 个展开的未提交文件已核对，
未删除或回退既有修改；另发现 3 个既有文件的并行前端修改和 6 个新增前端诊断文件，
已单独记录并保留，不归入本次 Rust 验收。相关 focus-stack npm 命令未变，冻结 Rust 源码未漂移。
v28 基线分支/二进制和 v31 冻结二进制未变，当前分支与空暂存区未变。
**9 张正对照回归通过；跨素材 N4 几何验收仍未通过。** 当前修改未提交或推送。

## 34. 跨机位尺度根因审计（2026-11-04，v33）

本轮新增只读诊断工具，保留 v32 的全部 500 个 Rust 文件和冻结二进制；没有修改任何生产门禁、
anchor/owner 选择、融合或缓存实现。目标是区分“单位/坐标换算错误”与“实测倍率不符合当前契约”，
而不是继续扩大搜索或把失败关系归一化成通过。

### 34.1 尺度门禁实际约束的量

`requirements.md` Requirement 2.1 要求锚点使用自身原生像素坐标；Requirement 6.3/6.9 明确要求
**两个 Capture_Station 的完整尺度比在 [0.95,1.05] 内**，越界时拒绝关系、不写入位姿。
Requirement 6.6 允许 8 自由度单应性，但不撤销 6.3/6.9 的范围限制。它不是“相对先验的残差尺度”。

生产 `station_relation_overlap_scale` 在全部实测内点的 source 重心处计算
`s = sqrt(abs(det(H) / (h20*x + h21*y + h22)^3))`，代表局部面积倍率的平方根。
canonical 方向由内容确定；方向改变会改变数值，不能把 1.1158 与 0.8921 当成相反的结论。
图像 Quality Gate 的 MTF、noise、boundary、来源证据等是后续门禁；前置关系不通过时不会运行。
把尺度除以被拒模型或先验尺度、改为残差尺度，即使仍写 [0.95,1.05]，也属于改变现有门禁含义。

### 34.2 采样、坐标与 anchor 核对

- 两套子集均为 scale=1；原始解码 8256×5504，光学校正后 8258×5507。代理长边 2400，
  `scale_factor=3.4408333333`，同一子集各源一致；原生匹配平面没有把 2400px 当成 8258px。
- 金字塔层使用 `native = factor*level + (factor-1)/2` 的像素中心映射，并在返回关系时共轭回
  原生坐标。共同下采样不改变尺度比，不能用代理倍率解释约 10% 的差异。
- `focus_anchor_render_frames` 使用 `A_station^-1 * G_source`，anchor 自身因此为恒等。
  六个实际融合 anchor 与 capture group 的 anchor 全部一致。由报告布局反解的 tile-local crop
  线性部分与恒等的误差不超过 2.3e-16，剩余只有平移，不存在隐藏的 0.9 倍 tile resize。
- 在当前组内成员上枚举候选 anchor，只作坐标诊断、不更改选择规则：阆苑 0→1/1→2 的面积尺度
  中位数范围分别为 0.90837–0.91419 / 0.91299–0.91643；文苑 0→2/1→2 为
  1.11544–1.12791 / 0.88385–0.89413。相关四对的 45 个成员组合均不能回到允许范围。
  枚举包含组内所有成员，不把候选身份自动视为合格融合成员。

### 34.3 不依赖拟合矩阵的对应点复核

只读工具从报告最多 64 个已保存实测对应点中选长边 500–3000px、最小高 200px 的非退化三角形，
直接计算 `sqrt(target_area/source_area)`，不读取拟合矩阵或报告 Local_Scale 来产生这一数值。
同时分四个空间区域统计，并保留直接点距比供对照。旋转/剪切会影响点距比，所以不能把点距比
中位数等同于生产的面积尺度。三角形只是对同批点的诊断组合，**不是新增内点或独立统计样本**。

| 机位对（实际 canonical 测量方向） | 生产尺度 | 对应点面积尺度中位数 | 四区域中位数范围 |
| --- | --- | --- | --- |
| 阆苑 0–1（1→0） | 0.913510 | 0.913848 | 0.91256–0.91444 |
| 阆苑 1–2（2→1） | 0.914201 | 0.914327 | 0.91388–0.91493 |
| 文苑 0–2（0→2） | 1.115817 | 1.115438 | 1.11409–1.11783 |
| 文苑 1–2（2→1） | 0.892051 | 0.891948 | 0.89068–0.89359 |
| 文苑正对照 0–1（0→1） | 1.000145 | 1.000176 | 0.99756–1.00209 |
| 9 张正对照 0–2（2→0） | 0.953599 | 0.953591 | 0.95337–0.95445 |

同一变换解析 Jacobian 的重算与报告采样 Local_Scale 的最大差异约 4.71e-6，远小于拒绝关系距离
阈值的差距。9 张正对照仍支持原有通过结论。不能把单个带噪三角形低于 0.95 当成新的拒绝条件：
生产规则考察的是拟合模型在内点重心的尺度，本工具不参与接受/拒绝判断。

另用冻结的 v32 二进制重新从 6 个真实 anchor NEF 经生产 RAW loader 开始做独立 pair probe，
不使用融合像素、station pose 或 world layout。阆苑两对获得 178 / 362 个 source 内点，文苑
三对获得 1027 / 324 / 259 个；这验证倍率并非只在融合后出现。该 probe 使用 2400px 分析图，
它是辅助 source 诊断，不替代原生 Virtual_Tile、Consensus_Feature 或最终 Quality Gate。

### 34.4 光学校正与物理成因的边界

六个 anchor 为同一机型、同一 `Nikkor Z 24-120mm f/4 S` 镜头；阆苑为 74mm、文苑为 70mm，
每套内部焦距一致。按仓库实际 Lensfun profile 的 70/85mm 标定点和生产 f32 插值方式核对参数，
再将原生 tile 对应点经 crop 和光学校正采样映射回传感器坐标。四条越界关系的面积尺度中位数
仍为约 0.91656、0.91979、1.10154、0.89741，不能通过关闭校正解释为 1:1。

这是依据实际光学坐标映射的**回投诊断**，不是重新匹配未校正 RAW，也不是新增关系证据。
同焦距不等于同成像倍率；现有元数据没有 FocusDistance/SubjectDistance，不能进一步断言由
相机距离、姿态、焦点呼吸中的哪一项单独造成。已确认的软件层原因是：不同机位的实测原生
像素倍率确有差异，而现有完整尺度契约将其拒绝。不能把它表述成“素材没有重叠”或“素材不可拼”。

### 34.5 开发交付与复验状态

新增 `tests/focus-stack-scale-audit.mjs` 和 `tests/support/station-scale-audit.mjs`，读取报告和
source alignment 缓存，输出带输入 SHA 的只读审计 JSON；拒绝覆盖输入或既有输出：

```sh
node tests/focus-stack-scale-audit.mjs /abs/Stack_Report.json /abs/source-alignment.json /abs/new-audit.json
node --test tests/focus-stack-scale-audit.test.mjs
```

7 项测试覆盖原生尺度正/负例、平移/旋转/剪切、数值 Jacobian、齐次矩阵归一化、anchor 消去、
共同比例/不等采样、退化/反射对应点和光学网格回投。三套 quality/datasets/regression 契约检查通过。
本轮不重复声称运行过 Rust 全库；生产源码和测试二进制逐文件相同，沿用 v32 的 1024 passed / 40 ignored。
两个完整 10 张子集已使用同一冻结二进制串行复验：

| 指标 | 阆苑 10 张 | 文苑 10 张 |
| --- | --- | --- |
| 连通分量 / accepted 关系 | 3 / 0 | 2 / 1 |
| Quality Gate / ROI | not_run / 0 | not_run / 0 |
| closure / TIFF | not_run / 未写出 | not_run / 未写出 |
| native wall（s） | 385.688 | 396.378 |
| OS maximum RSS（B） | 9,115,795,456 | 8,685,355,008 |

input、grouping、intra-station、fusion、station relations、closure 和 Quality Gate 的非计时内容
均与 v32 完全相同；每套 9 个原生/质量/Coverage PNG 的 SHA 也相同，真实 RAW 输入 SHA 不变。
运行耗时仅作记录，本轮不宣称性能改善。外层均退出 1、Rust child 均退出 101，保持拒绝语义。

9 张 NEF 本轮没有重复冷/热运行；已重新校验冻结二进制、500 个 Rust 源码文件、v32 冷/热报告
及两个 TIFF 的实际 SHA，原有 pass / 8 ROI 和输出 SHA 保持有效。另将其保存的实测点作为此次
只读审计的正对照，未改动 owner/source 证据或基线。新工具源码 3 个文件，生产 Rust 0 改动。

本地证据位于 `output/acceptance/cross-station-v33-scale-audit/`，包含 RAW anchor probe、
完整复跑、坐标/采样审计、光学回投及测试日志；汇总见
[v33 尺度根因审计与复验](benchmarks/cross-station-v33-scale-root-cause.json)。
工作区开始时 274 个未提交文件的清单/补丁已保存；期间观察到 2 个并行文件修改和 3 个前端
临时诊断文件删除，均按并行工作原状保留，不覆盖、不恢复，不归入本轮验收。
当前 HEAD、v28 基线分支、空暂存区未变，未提交或推送。

**根因审计不等于断连修复。** 当前没有找到满足原生完整尺度契约的合格连接，不能在不改变该契约
的同时把这些已测模型改判为 accepted。N4 保持未完成。后续若产品要求覆盖这些倍率变化，必须
先单独设计并确认独立标定的采样网格及几何验收契约，完整保留 RAW fidelity、holdout、质量与
来源门禁；这会涉及验收语义，不属于本轮获准的纯坐标修复，未在生产实现。

## 35. v34 跨机位倍率几何契约（2026-11-04）

本轮用户已明确授权设计并实施跨机位倍率支持，承接§34的根因结论。详细分阶段方案见 [跨机位倍率方案](focus-stack-variable-magnification.md)。只扩展经原生证据验证的几何关系，不修改画质门禁、来源证据和已验收9张基线。本轮结果与保留的失败阶段如下。


- **方案/实现**：保留 `[0.95,1.05]` 旧路径及所有修复增量常量；仅Virtual Tile增加`bounded_native_magnification_v1`（0.8–1.25包络、空间留出、四区域、局部Jacobian重复性/形变约束）。候选和最终solver共用证据计算，报告新增可选`scale_evidence`；RAW seed仍非权威。需求6.11及设计文档已同步。
- **验证**：新增4项定向测试通过；release全库1028 passed / 40 ignored / 0 failed，fmt/check、3项Node contract和7项尺度审计测试通过。Quality Gate生产代码、owner/source采样和缓存实现未改动。
- **阆苑10**：连通3→1、接受2条关系、8 ROI/8 station owner evidence，density 1.24609375；local-scale/gradient/noise/色差/owner sharpness通过，但有效覆盖0.97192542<0.98，QG fail。10147×12663原生TIFF已写出，不能标为验收通过；PS参考配准另有失败。native wall686.145s、RSS8.841GB、RAW20/fusion6。
- **文苑10**：连通2→1，三个局部关系先通过，闭环拒绝0–2后保留2条；直接P95 16.233545px，经density 1.1787109375换算19.134657px，原门禁拒绝输出。QG not_run/0ROI、无TIFF。native wall244.757s、RSS9.327GB；失败时fusion统计未完整汇总，不能把报告0解释为无融合开销。
- **来源与对照**：两套各9张原生gray/quality/coverage PNG的SHA与v32完全一致，输入与参考文件未改。9张NEF冷/热均pass/8ROI、RAW18/18、fusion6/6、source/native hits113/15；TIFF与v28/v31/v32相同，每次12个非计时证据部分与v32一致。新冷/热wall401.591/376.232s、RSS9.707/9.513GB，默认预算与backing门禁通过。独立进程未清OS page cache，单次时间波动不宣称显著加速。
- **未完成边界**：本轮完成倍率契约实现和两套真实复验，跨素材完整验收仍未完成。下一步先做文苑三关系的联合原生几何一致性，再核对阆苑有效来源footprint/alpha支持及参考注册；不放宽闭环或98%覆盖门槛。树图`no_constraints`不解释为零闭环误差。

机器摘要：[v34倍率契约与复验](benchmarks/cross-station-v34-variable-scale-acceptance.json)。阶段方案、阈值和逐关系数值见[完整方案](focus-stack-variable-magnification.md)。

阆苑完成全图子集预览和四个原生512×512裁切观察，右边界透明缺口保留，完整验收仍失败；视觉记录见v34输出目录`visual/manifest.json`。9张1:1视觉证据按相同TIFF SHA继承。冻结候选SHA：`f0b24b58e7e505bce98636b45f8da2cbfe14d4f400efee31c79da76d7d865552`（501个Rust文件）。


## 36. v35：联合闭环与有效来源域（2026-11-04）

本阶段按文苑联合原生几何→阆苑覆盖/参考配准→同一9张NEF正对照执行。文苑完整验收通过；阆苑QG与参考配准通过，但原生验收仍因树图`no_constraints`拒绝。N4保持未完成，v28/v31/v32/v34冻结基线和其他未提交修改保留。

文苑在剪边前尝试已有原生证据支持的联合优化，并将几何修正与输出采样密度分开，保留三条关系。原生每对P95从16.233545降至1.810738px，density1.1796875对应输出P95=2.136105px；既有每对3px、median2px及局部/corner门禁不变。

覆盖统计改用完整原生逆映射得到的独立来源域，排除校准画布padding；分母不读取alpha/owner/亮度，域外alpha不能补偿域内孔洞。98%阈值、其他画质判据及来源证据保留。Composition旧计数不变，新增QG审计保存旧矩形/全部alpha与实际域内/域外计数。逐行计算，不增加RAW解码或全画布域位图。

| 最终10张运行 | 物理覆盖率 | QG / ROI | 原生wall | 峰值RSS bytes | RAW / fusion | 完整原生验收 |
| --- | --- | --- | --- | --- | --- | --- |
| 文苑 | 99.472410% | pass /8 | 634.717s | 9,101,803,520 | 20 /6 | 通过 |
| 阆苑 | 99.441301% | pass /8 | 496.666s | 9,656,811,520 | 20 /6 | 拒绝：树图无独立闭环 |

两套均保留8份station owner evidence。精确覆盖计算耗时2.592/2.554s；修正前后TIFF逐字节不变（文苑对比本轮几何候选，阆苑对比v34）。参考注册采用受限尺度/窗口搜索，保留原整幅2048坐标格的3px/40-inlier门禁，加入独立留出、多区域支持与禁止域外外推。阆苑最终114个留出inlier、P95=1.910355px，验证域27.3995%，仅52个内容tile做PS辅助比较；不能称整幅参考验收或清晰度优于PS。

阆苑0–2候选未通过既有原生证据，不能补成虚假闭环。下一步审计真实重叠与独立关系证据；若确实只有树图，需单独明确其独立几何验证合同，不能静默放宽当前converged要求。旧degraded/not_applicable、原生透明边界及失败日志保留。

完整实现、坐标单位、覆盖审计和视觉边界见[跨机位倍率方案](focus-stack-variable-magnification.md#v35联合原生几何有效来源覆盖和参考配准2026-11-04)。冻结候选SHA `392f410cf9d43e2dfe3dc8a4c399fbe9997f15ae495e023dc08c4a602a3f331a`；fmt/check、3项Node契约、6项定向测试通过，完整release库测试1035 passed /41 ignored /0 failed（280.08s）。

### 同一9张默认缓存冷/热回归

DSC6479–6487、scale=1，默认512MiB source /1GiB native缓存及8GiB/4GiB backing；每轮独立进程与运行内缓存，未清空OS页缓存。“冷/热”不等于磁盘冷启动。

| 指标 | v35冷 | v35热 |
| --- | --- | --- |
| QG / ROI / station owner evidence | pass /8 /8 | pass /8 /8 |
| RAW解码 / native fusion | 18 /6 | 18 /6 |
| source / native cache hits | 113 /15 | 113 /15 |
| 原生进程wall | 532.148s | 677.405s |
| 峰值RSS（bytes） | 8,794,636,288 | 9,421,586,432 |
| fusion累计时间 | 325.444s | 409.228s |
| 精确覆盖计算 | 2.407s | 2.094s |

两轮物理有效覆盖97,847,414/97,857,864=99.989321%；域内缺失10,450、域外不透明11,357，旧矩形98,588,607/全部alpha97,858,771保留。所有其他QG判据、Composition及11个非QG证据部分与旧基线一致，冷/热归一化业务报告一致。缓存预算/复用检查通过、backing无失败。15897×6720 RGBA16 sRGB TIFF的SHA仍为`718a903dcd0bf102fd7ea7f2be652168ad972f18bc540f5854225ee2f1bb47e3`，按相同字节继承原1:1视觉证据。

**这表示画质、证据和缓存行为回归通过，不表示性能无回退。** v34冷/热wall为401.591/376.232s，本轮分别增加32.51%/80.05%；热跑比冷跑慢。精确覆盖新增约2秒不能解释全部差异，融合累计wall也由254.915/233.333s增至325.444/409.228s。单次运行尚未隔离系统负载、IO与代码成本，不能归因为其中任一因素，也不能宣称加速；后续需同机受控交错运行冻结v34/v35、重复测量并分析阶段CPU/IO。原始慢跑数据保留，不用择优重跑覆盖。

29张原始输入、503个Rust源码及构建输入、v28/v31/v32/v34冻结二进制和v28分支均重新核验未变。工作区其他修改保留，未提交或推送。机器可读结果见[最终验收摘要](benchmarks/cross-station-v35-native-domain-acceptance.json)：状态为`partial_acceptance`，阆苑完整原生验收仍失败。局部画质回归通过不能勾选N4全项。
