# Implementation Plan: 分层机位组景深与拼接改造

## Overview

本计划是对 `src-tauri/src/panorama_stitching.rs`、`panorama_utils/{stitching,mosaic,seam_cut,photometric,registration,mosaic_diagnostics}.rs`
与 `src-tauri/src/image_stack.rs` 的**增量改造**，实现语言为 Rust（前端契约测试为 Node/JS）。
不新建并行流程：所有新增类型落在新模块 `src-tauri/src/panorama_utils/stack_pipeline/` 下，
行为改动全部就地修改现有函数与常量。

执行顺序严格遵循 design.md「分阶段落地顺序」的阶段 0→8 依赖链。
贯穿全程的两条纪律：

- **先加观测，再改行为**：Stack_Report 与 Quality_Gate 先以「记录但不阻止」模式上线，
  取得阆苑女仙 84 张的实测数字，再单独用一个任务收紧阈值/切换为阻止模式。
- **旧路径保持可编译可对照**：`focus_stack_stitcher`、`progressive_seam_stitcher`、
  `detail_preserving_mosaic`、`crop_to_valid_rectangle`、`sharpen_focus_tile_detail`
  的函数本体不改行为，只切走默认路径的调用点，避免 `tests/focus-stack-quality-contract.mjs`
  的源码级断言与 `mosaic.rs` 的 24 个单元测试失效。

测试分三层，任务里明确区分：

- **常规回归（不读 RAW、不联网）**：`cargo test --lib` 的 `proptest`（`ProptestConfig { cases: 100, .. }`）
  与 `tests/*.mjs` 的源码/常量级断言。
- **Acceptance_Harness（需要阆苑女仙 84 张 NEF）**：`#[ignore]` 标记，手动触发。
- **集成测试（1–3 个例子）**：时限类判据。

## Tasks

- [x] 1. 阶段 0：一致性清理与观测骨架（无行为变更）

  - [x] 1.1 把遗留的 200 统一到 500，修复当前失败的前端契约断言
    - 修改 `tests/image-stack-preview-contract.mjs:47`：
      `assert.equal(frontendMaxSources, 200, ...)` 改为断言 `500`，并把消息改为 500 的表述
    - 修改 `README.md:369` 的「可一次选择 2–200 张」为「2–500 张」
    - 不修改 `src/utils/imageStackPipeline.ts` 与 `src-tauri/src/image_stack.rs:33`
      的 `IMAGE_STACK_MAX_SOURCES`（已是 500），保持
      `frontendMaxSources === backendMaxSources === stitchingMaxSources`
      三向一致断言原样通过
    - 运行 `node tests/image-stack-preview-contract.mjs` 确认该测试由失败转为通过
    - _Requirements: 1.8_

  - [x] 1.2 新增 `STACK_PIPELINE_VERSION` 与 `report.rs` 的 Stack_Report 结构
    - 新建 `src-tauri/src/panorama_utils/stack_pipeline/mod.rs` 与 `report.rs`
    - 在 `image_stack.rs` 中与现有 `IMAGE_STACK_PIPELINE_VERSION` 并列声明
      `STACK_PIPELINE_VERSION`
    - 按 design.md「Stack_Report 的 JSON schema」定义全部顶层字段（`input`/`grouping`/
      `intra_station`/`fusion`/`virtual_tiles`/`topology`/`station_relations`/`closure`/
      `residual_warp`/`tone`/`composition`/`quality_gate`/`output`/`resources`/`degradation`），
      缺省值显式写出，用 `serde` 序列化
    - 在 `stitch_images_with_options` 的 `BlendMode::FocusStack` 分支的全部终止路径
      （成功、降级、拒绝、取消）写出报告 JSON，30 秒内完成；此阶段**不做任何判定**
    - _Requirements: 10.8, 11.16_

  - [x] 1.3 新增 `degradation.rs` 的失败标识符全集与 `DegradationLedger`
    - 按 design.md「稳定机器可读失败原因标识符」全表以 `const &'static str` 声明
      全部标识符（`input_*`/`source_decode_failed`/`grouping_*`/`intra_station_*`/`fusion_*`/
      `cache_*`/`topology_*`/`station_relation_*`/`station_pose_*`/`geometry_disconnected`/
      `closure_*`/`residual_warp_*`/`tone_*`/`composition_*`/`canvas_long_side_exceeded`/
      `output_*`/`quality_gate_*`/`memory_threshold_exceeded`/`run_cancelled_by_user`/
      `diagnostics_*`）以及不可测量原因标识符集合
    - 实现 `DegradationLedger` / `DegradationEntry { reason, severity, detail }`
    - 把现有分散的 `println!` 诊断点**并行**写入 ledger，保留原 `println!` 行，
      **不改变任何现有回退行为**
    - _Requirements: 12.7, 12.9_

  - [x] 1.4 扩展 `tests/focus-stack-quality-contract.mjs` 的标识符与 schema 断言
    - 新增失败标识符全集格式断言 `^[a-z0-9_]{1,64}$` 与唯一性断言
    - 新增 Stack_Report 顶层字段存在性断言（源码级读取 `report.rs`）
    - 保留现有 `SELECTION_LONG_SIDE >= 512`、`OWNERSHIP_MISMATCH_PENALTY >= 1`
      等既有断言不变
    - _Requirements: 15.9_

  - [x]* 1.5 属性测试：失败标识符格式稳定
    - **Property 79: 失败原因标识符格式稳定**
    - **Validates: Requirements 12.7**
    - 生成器 `arb_stack_report()`

- [x] 2. 阶段 0 检查点
  - 确保所有测试通过（含此前失败的 `tests/image-stack-preview-contract.mjs`），
    Acceptance_Harness 之外的套件不出现回归，遇到问题询问用户。

- [x] 3. 阶段 1：确定性基线

  - [x] 3.1 消除 `HashMap` 迭代顺序依赖
    - 在 `stack_pipeline` 中新增 `sorted_pairs(&map)` 辅助函数
    - 改写 `solve_focus_capture_group_poses()` 中 `for (&(i, j), m) in matches`
      等直接迭代 `HashMap`/`HashSet` 的位置，改为先 `collect` 到 `Vec` 再按稳定键排序
    - 新增排序一律使用 `f64::total_cmp` + 稳定次级键
    - _Requirements: 14.6_

  - [x] 3.2 把影响判定与输出像素的浮点归约改为确定性归约
    - 把 `rayon` 的 `par_iter().sum()` 类归约替换为：≤4096 项串行 `fold`；
      大规模按编译期常量块大小分块并行求部分和、再按块索引串行累加
    - `par_chunks_mut` 的按目标索引分区写入保持不变
    - _Requirements: 14.6_

  - [x] 3.3 RANSAC 随机种子改为从输入内容派生
    - 种子取 `SHA-256(cache_key)` 前 8 字节，移除系统时间与线程 id 依赖
    - 在 Stack_Report 的 `resources` 段记录线程数与种子来源
    - _Requirements: 14.6_

  - [x] 3.4 `focus_source_order_group_compositor_order()` 的 tie-break 去文件名化
    - 把 `natural_path_cmp` tie-break 暂改为世界坐标排序（阶段 4 引入行列索引后再切换）
    - 同步更新依赖该顺序的现有单元测试断言
    - _Requirements: 14.6_

  - [x]* 3.5 属性测试：输出逐字节可复现
    - **Property 90: 输出逐字节可复现**
    - **Validates: Requirements 14.6**

  - [x]* 3.6 属性测试：分组与拓扑对重命名与导入顺序不变
    - **Property 2: 分组与拓扑对重命名与导入顺序不变**
    - **Validates: Requirements 1.2, 1.3, 5.8**
    - 生成器 `arb_scan_grid(plane)`
    - 执行记录：重放 `grouping-flaky-seed.txt` 中的 `cc a27d43fcfd157267ab58050680837905edd56bbbad8644ff68c2f3b11d361768`
      在 13d0af6b 与 c04687cf 均通过；失败来自并行属性测试共享 run sink 的交错。生产入口与该属性现在
      用 run scope 串行化 reset/render/snapshot，未改变分组算法。

  - [x] 3.7 新增确定性回归断言到 `tests/focus-stack-quality-contract.mjs`
    - 新增「`stack_pipeline` 模块内不出现对 `HashMap` 的直接 `for ... in` 迭代」断言
    - _Requirements: 15.9_

- [x] 4. 阶段 1 检查点
  - 验证同输入在 1/2/8/16 线程下输出 SHA-256 相同；确保所有测试通过，
    遇到问题询问用户。

- [ ] 5. 阶段 2：Virtual_Tile 结构与缓存

  - [x] 5.1 扩展 `FocusVirtualTile` 为 `VirtualTile`
    - 在 `stitching.rs` 就地扩展现有 `FocusVirtualTile`：`image` → `pixels`、
      `group_index` → `station_index`、`source_ids: Vec<usize>` →
      `provenance: Vec<SourceProvenance>`
    - 新增 `ownership: OwnershipMap`（`Vec<u16>`，0 = `NO_OWNER`，带 `legend: Vec<PathBuf>`）、
      `sharpness_confidence: ConfidenceMap`、`coverage: CoverageMask`、
      `color_encoding: ColorEncoding`、`width` / `height`
    - 移除 `FocusVirtualTile` 与 `focus_stack_virtual_tiles()` 的 `#[allow(dead_code)]`，
      接入生产路径；保持 `focus_stack_virtual_tile_geometry()` 签名向后兼容
    - 构造函数强制不变式：`provenance.len()` 等于参与合成源图数、
      `Σ owned_pixels` 等于已赋 owner 像素数、coverage 与 ownership 双向一致、
      `pixels` 有效范围等于 coverage 联合边界
    - _Requirements: 4.1, 4.2, 4.6, 4.7_

  - [x] 5.2 让 `focus_stack_stitcher_unfilled()` 暴露 Coverage_Mask 与 Ownership_Map
    - 该函数已不填充投影梯形外的像素，把其内部掩膜作为返回值暴露出来，
      不改变像素写入行为
    - 同步更新 `stitching.rs` 中该函数现有调用点与相关单元测试断言
    - _Requirements: 3.7, 3.11, 4.6_

  - [x] 5.3 实现 `virtual_tile.rs` 的 `Virtual_Tile_Store` 磁盘缓存
    - 目录布局 `<app_cache_dir>/stack-virtual-tiles/v1/<cache_key>/`，
      复用 `image_stack.rs:697` 的 `app_cache_dir()` 用法
    - `cache_key = SHA-256(STACK_PIPELINE_VERSION ‖ sorted_paths ‖ sorted_source_sha256)`，
      三要素独立写入 `meta.json` 供读取时逐项复核
    - 无损载荷：`pixels.f32.zst` / `ownership.u16.zst` / `coverage.bits.zst` /
      `confidence.f16.zst`，不使用任何图像编码器
    - 原子写入：先写 `.tmp-<uuid>/`、`fsync`、再 `rename`
    - `index.json` 记录 `last_access_epoch_ms` 与 `bytes`，超 64 GiB 按最后访问时间
      从早到晚整条删除
    - 损坏检测（字段缺失 / 尺寸不符 / SHA-256 无法复核）→ 删除条目、重新合成、
      记录 `cache_entry_*` 标识符
    - 写入失败（空间不足 / 目录不可写）→ 保留内存瓦片、不中断、标记 `cache_write_unavailable`
    - _Requirements: 4.3, 4.4, 4.5, 4.9, 4.10, 4.11_

  - [x] 5.4 实现 Source_RAW 只读访问与运行前后 SHA-256 复核
    - 全部源文件用 `File::open` 只读打开；成功、失败、取消三条结束路径都重算 SHA-256
      并与读取前比较，写入 Stack_Report 的 `input.sources[*].sha256_before/after`
    - _Requirements: 4.8, 12.8_

  - [x] 5.5 实现 `Virtual_Tile_Store::lease()` 租约计数器
    - RAII 租约，上限 2 个完整尺寸 Virtual_Tile；超限先淘汰最久未用租约，
      无可淘汰租约时返回错误
    - 观测到的最大同时租约数写入 `resources.max_resident_virtual_tiles`
    - _Requirements: 14.4_

  - [x]* 5.6 属性测试：覆盖与 ownership 双向一致
    - **Property 13: 覆盖与 ownership 双向一致**
    - **Validates: Requirements 3.7, 3.11**
    - 生成器 `arb_coverage_shape()`

  - [x]* 5.7 属性测试：Virtual_Tile 溯源求和恒等
    - **Property 18: Virtual_Tile 溯源求和恒等**
    - **Validates: Requirements 4.1**
    - 生成器 `arb_virtual_tile()`

  - [x]* 5.8 属性测试：Virtual_Tile 缓存无损往返
    - **Property 19: Virtual_Tile 缓存无损往返**
    - **Validates: Requirements 4.3**

  - [x]* 5.9 属性测试：缓存键三要素判定命中
    - **Property 20: 缓存键三要素判定命中**
    - **Validates: Requirements 4.5, 4.9**

  - [x]* 5.10 属性测试：缓存淘汰保持容量上限与访问时序
    - **Property 21: 缓存淘汰保持容量上限与访问时序**
    - **Validates: Requirements 4.4**

  - [x]* 5.11 属性测试：缓存损坏必然被识别为未命中
    - **Property 22: 缓存损坏必然被识别为未命中**
    - **Validates: Requirements 4.10**

  - [x]* 5.12 属性测试：掩膜与像素同构
    - **Property 23: 掩膜与像素同构**
    - **Validates: Requirements 4.6, 4.7**

  - [x]* 5.13 属性测试：Source_RAW 在任何路径下保持不变
    - **Property 24: Source_RAW 在任何路径下保持不变**
    - **Validates: Requirements 4.8, 12.8, 13.4**

  - [x]* 5.14 属性测试：同时常驻 Virtual_Tile 不超过 2 个
    - **Property 88: 同时常驻 Virtual_Tile 不超过 2 个**
    - **Validates: Requirements 14.4**

  - [x]* 5.15 单元测试：像素类型与缓存降级注入
    - 断言 `pixels` 为 `Rgb32FImage` 且 `color_encoding` 标识存在（需求 4.2）
    - 注入只读缓存目录，断言 `cache_write_unavailable` 且运行不中断（需求 4.11）
    - _Requirements: 4.2, 4.11_

- [x] 6. 阶段 2 检查点
  - 验证缓存往返逐元素相同、同输入在命中/未命中两种状态下输出 SHA-256 相同；
    确保所有测试通过，遇到问题询问用户。

- [ ] 7. 阶段 3：机位层质量（分组 + 配准 + 合成）

  - [x] 7.1 收紧 `focus_match_is_capture_station_link()` 为需求 1.1 的四项证据
    - 在 `panorama_stitching.rs` 就地改造：内点门槛
      `FOCUS_LOCAL_MODEL_MIN_INLIERS = 6`（`panorama_stitching.rs:48`）提到 30；
      把 `FOCUS_BRACKET_MIN_OVERLAP_SUPPORT = 0.60`（重叠面积比）与
      `focus_overlap_quality()` 的 `intensity_ncc ≥ 0.45` 区分开，NCC 门槛提到 0.60，
      面积支持改用 `panorama_spatial_support()` 的内点凸包比 ≥ 0.20
    - 新增尺度比判据：取单应性左上 2×2 子块奇异值几何均值 `sqrt(σ1·σ2) ∈ [0.98, 1.02]`
      （与需求 6.3 共用同一函数，仅阈值不同）
    - 显式化 `StationEvidence { inliers, overlap_ncc, spatial_support, scale_ratio, score }`
    - 同步更新 `tests/focus-stack-quality-contract.mjs` 中涉及这些常量的源码级断言
    - _Requirements: 1.1_

  - [x] 7.2 把 `split_focus_local_component_by_motion()` 改为累计位移拆分
    - 现有判据是相邻候选中心位移，阈值 `FOCUS_BRACKET_MAX_CENTER_MOTION_RATIO = 0.075`
      （`panorama_stitching.rs:67`）；改为沿候选顺序的**累计**中心位移，阈值收紧到 `0.02`
    - 候选顺序由世界中心排序给出，`natural_path_cmp` 仅在世界中心 `total_cmp` 相等时生效；
      判定分数 tie-break 仅在两候选 `score` 之差 ≤ 0.001 时才允许用绝对路径
    - 复用 `FOCUS_BRACKET_MAX_COMPONENT_SOURCES = 48`，超限时在累计位移最大处反复拆分，
      每次拆分写入 `grouping.splits` 并记录 `grouping_member_limit_split`
    - 同步更新该函数现有单元测试与常量断言
    - _Requirements: 1.2, 1.4, 1.5, 1.10_

  - [x] 7.3 补齐孤立源图、解码失败与分组上报
    - `focus_local_bracket_components()` 中当前被静默跳过的孤立/不可解码源图，
      改为记录到 `GroupingOutcome::{isolated, undecodable}`，
      带文件绝对路径与 `grouping_no_overlap_evidence` / `source_decode_failed` 标识符，
      且不影响其余源图的分组结果
    - 把每个 Capture_Station 的成员路径、判定依据类型、判定分数、内点数、
      内点空间支持比例写入 `grouping.stations`
    - 补一条断言防止 `focus_graph_capture_sequence_boost()` 的
      `FOCUS_BRACKET_MAX_CAPTURE_GAP` 加权回归到参与成员判定
    - _Requirements: 1.6, 1.7, 1.9_

  - [x] 7.4 实现 `intra_station.rs` 的锚点帧选择与两级配准预算
    - 锚点帧从 `focus_capture_groups()` 现有的 `let anchor = members[0]`（世界中心排序第一）
      改为「有效像素 Sharpness_Score 中位数最高者，差 < 0.01 时按绝对路径升序取第一」，
      锚点帧 `member_to_anchor` 设为恒等
    - 新增 `INTRA_STATION_ANALYSIS_LONG_SIDE = 1400` 作为默认路径实际分析长边配置，另以
      `INTRA_STATION_ANALYSIS_MAX_LONG_SIDE = 2048` 表示需求硬上限；不再复用
      `scalable_alignment_budget()` 的 2400/1800/1536（那是机位间预算），并把实际配置值 1400
      原样写入 Stack_Report；在该分辨率执行 2 轮全局 + 局部配准后再进原生 patch refinement
    - _Requirements: 2.1, 2.2_

  - [x] 7.5 扩展 `refine_native_layer()` 的控制点判据与逐帧回退
    - 保留 `NATIVE_REFINE_STEP = 16`（`mosaic.rs:50`）；匹配窗口与搜索半径按实测定为
      `INTRA_STATION_MATCH_RADIUS = 25`、`INTRA_STATION_SEARCH_SAMPLES = 7`、
      `INTRA_STATION_PATCH_HALF = 33`（67×67 缓冲），即最细一遍窗口边长 51 原生像素、
      `spacing = 2.0` 一遍搜索可达 `INTRA_STATION_SEARCH_RADIUS = 64` 原生像素，
      两者均在需求 2.3 的「窗口 ≥32、可达 ≤64」之内；旧 mosaic 对照路径保留
      `LEGACY_MOSAIC_PATCH_HALF = 28` 与半径 10，输出逐字节不变
      （实测依据：33 原生像素窗口在绢底稀疏笔画上结构不足，相关中位数 0.60 对
      密绘机位的 0.84；51 原生像素窗口面积为 2.4 倍，7 帧全部通过需求 2.8）
    - **新增焦平面匹配低通**（`intra_station::match_focal_plane`）：机位内两侧本就聚焦
      在不同平面，原始灰度 NCC 无法把同一内容的清晰版与失焦版看成同一内容
      （实测接受率 0.02–0.4%）。两侧先各做一次 `variance = 0.5` 的共同低通压掉互不相关的
      颗粒，再用自适应步长（`MIN_STEP = 0.01`、`MAX_STEP = 0.5`、`STEP_GROWTH = 1.6`、
      预算 `MAX_VARIANCE = 16.0`）把较清晰一侧低通到与较模糊一侧的二阶差分能量对数距离
      最近处；固定一遍 `[1 2 1]/4` 步长实测会在第一步就越过配平，必须可回退可细分。
      实测两侧能量比中位数由 0.39–0.89 提升到 0.96–0.99
    - 双向一致性阈值取需求 2.4 规定的 1.0 原生像素（原 0.40 是机位**间**路径的遗留值，
      低于失焦帧的可定位精度）；相关性门槛按实测拐点定为
      `INTRA_STATION_MIN_CORRELATION = 0.70`（该桶双向闭合率 79%，0.65 桶降到 69%；
      0.90/0.95 桶反而降到 87%/73%，因此不向上抬），并删掉原先额外的、无需求依据的
      0.90 二重门槛；`INTRA_STATION_MIN_PEAK_MARGIN` 沿用匹配器一直使用的 0.006
      （实测各帧 p10 为 0.011–0.145，该门不成为约束）。实测分布逐遍写入运行日志
    - 新增 8 邻域中位数一致性检查（差 > 4.0 原生像素则拒绝该控制点）
    - 新增对称重投影误差门（> 0.01×锚点帧长边则拒绝，该位置保留全局模型结果，
      计入 `rejected_control_point_ratio`）
    - 新增逐帧回退：patch refinement 后中位对称重投影误差不低于全局模型结果时，
      整帧丢弃局部场，`status = GlobalFallback`，标识符 `intra_station_local_fallback`
    - **需求 2.7 的空基线必须是第三种结论**：粗遍（`spacing = 2.0`）接受点不足时
      `global_errors` 为空、`median_of(&[])` 返回 `0.0`，比较退化为「无条件回退」。
      改为①基线取第一个**真正改动过位移场**之前、且样本数 ≥
      `INTRA_STATION_MIN_BASELINE_SAMPLES = 6` 的那一遍；②任一侧样本不足时判为
      `LocalFieldVerdict::Indeterminate`，保留实测位移场、不回退，写入
      `intra_station[*].local_field_verdict` 与标识符
      `intra_station_local_baseline_indeterminate`
    - 新增 `inlier_area_coverage`（含已接受局部匹配的控制点单元面积和 / 锚点帧有效像素面积），
      < 0.20 时 `status = Failed`，标识符 `intra_station_registration_failed`；
      该面积取**两遍 spacing 的并集**（`ControlPointGrid::absorb`），因为两遍接受的匹配都进入
      同一个位移场，只算最后一遍会低估需求 2.8 的定义；需求 2.5 的邻域门仍按单遍判定
    - 在机位内路径上取消 `if sampler.scale <= 2.0 { return; }` 早退，
      该早退保留在旧 mosaic 对照路径
    - 把 `anchor_path`、每帧变换、内点数、覆盖率、实测控制点间距、被拒控制点比例、
      中位误差、全局基线中位误差与样本数、局部样本数、需求 2.7 结论、
      焦平面匹配前后的能量比、状态写入 `intra_station[*]`
    - 同步更新 `mosaic.rs` 中受影响的现有单元测试（全部保持通过）
    - 实测（`langyuan-10` / `wenyuan-10`，两套素材各 10 帧 3 机位）：7 个非锚点帧全部
      `Local`，覆盖率 0.296–0.842 / 0.332–0.696，需求 2.7 结论全为 `kept`，
      3 个机位全部走 GraphCut，无 `SingleFrameDegraded`、无
      `intra_station_registration_failed`
    - _Requirements: 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9_

  - [x] 7.6 Sharpness_Score 归一化与 32px 采样窗口
    - 落地位置是 `stack_pipeline/focus_fuser.rs` 而非 `mosaic.rs`：默认路径是
      `layered_virtual_tile_compositor`，每个虚拟瓦片只含 1 个机位
      （`capture_group_count == 1` → `shifted_mosaic` 恒假），`mosaic.rs` 的
      `acutance` / `cell_focus` / `ownership_grid` 只服务旧对照路径；
      新模块复用这些原语（`acutance_with_step` / `cell_focus_with_step` /
      `cell_probe_offsets`）但不调用 `detail_preserving_mosaic`，避免其 cell-wise
      局部形变打碎不同焦深上有意存在的笔画
    - `mosaic.rs::acutance()` 当前返回无界比值：新增常量 `SHARPNESS_HALF_SCALE`，
      归一化为 `a / (a + SHARPNESS_HALF_SCALE)`（保序，不改变任何 owner 选择）
    - `cell_focus()` 保持中心 + 四角 5 探针，把 13×13 网格点间距改为
      `max(1, round(32 / 13))`，使有效窗口边长约 32 原生像素，梯度仍是相隔 4 个采样步长
    - 为「候选与当前底图使用相同采样位置与相同窗口尺寸」补显式断言
      （现有 `ownership_grid()` 已把同一 `(ax, ay, cell_size)` 传给两侧）
    - _Requirements: 3.2, 3.9_

  - [x] 7.7 ownership 单元尺寸与不一致性度量
    - 同上落地在 `stack_pipeline/focus_fuser.rs`（`mosaic.rs` 的
      `ownership_disagreement` 保留旧的高分位数语义给对照路径，中位数是新模块的
      独立归约）；单元尺寸用 `floor` 而非 `ceil`：`ceil(9504/512)=19px` 只给
      501 个单元，低于需求 3.1 的 512 单元下界，`floor` 得 18px / 528 单元，
      两条判据同时成立
    - `cell_size = clamp(ceil(long_side / SELECTION_LONG_SIDE), 8, 64)`，
      复用 `SELECTION_LONG_SIDE = 512`（`mosaic.rs:25`）
    - `ownership_disagreement()` 从「保留高分位数」改为需求规定的
      「低通后像素差绝对值的**中位数**」并归一化到 `[0, 1]`
    - 惩罚沿用 `OWNERSHIP_MISMATCH_PENALTY = 2.4`（`mosaic.rs:52`，≥1.0），
      仅在 `disagreement > 0.2 且候选 Sharpness_Score 更高` 时施加
    - 同步更新 `tests/focus-stack-quality-contract.mjs` 的常量推导断言
      （`ownership cell ∈ [8, 64]`）
    - _Requirements: 3.1, 3.3_

  - [x] 7.8 实现 `focus_fuser.rs` 的多标签图割
    - 按确定性帧顺序（锚点帧首位，其余按中位 Sharpness_Score 降序，同值按绝对路径升序）
      反复调用 `seam_cut::cut_grid(width, height, preference, disagreement, fixed)`
      做「当前底图 vs 下一候选」的二值割
    - `preference[i] = data_cost(current) − data_cost(candidate)`，
      `data_cost = (1 − normalized_sharpness) + mismatch_penalty`；
      平滑项沿用 `cut_grid` 内部的
      `OWNERSHIP_PAIRWISE_BASE + (d_i + d_j)·OWNERSHIP_PAIRWISE_DISAGREEMENT_WEIGHT`
    - `fixed[i]` 把仅当前 owner 覆盖的单元钉在当前 owner、仅候选覆盖的单元钉在候选
    - **不改动 `cut_grid` 本体的最小割语义**，保持其与穷举最小值对齐的现有测试通过
    - _Requirements: 3.4_

  - [x] 7.9 图割超时降级与单帧机位短路
    - 多标签循环外挂单调时钟，120 秒超时后剩余候选改为逐单元取最低代价，
      `solver_status = PerCellFallback`，标识符 `fusion_graph_cut_timeout`，
      并把 `graph_cut_seconds` 写入报告
    - `members.len() == 1` 或仅 1 帧配准成功时跳过图割，`solver_status = SingleFrame`，
      全部已覆盖像素指向该源图
    - _Requirements: 3.5, 3.10_

  - [x] 7.10 ownership 硬归属像素写入
    - 像素写入阶段严格按 `owner` 复制被选中源图像素，不做 owner 之间任何频带加权平均，
      不写过渡混合带
    - 移除机位层对 `STREAMING_OWNERSHIP_FEATHER = 0.65`（`mosaic.rs:35`，
      用于 `mosaic.rs:2521` 的 `imageops::blur`）的使用，改为在 ownership 网格分辨率上硬边；
      该常量与旧对照路径的用法保留
    - _Requirements: 3.6_

  - [x] 7.11 Sharpness_Confidence 与低清晰度区域
    - `conf = clamp((s_best − s_second) / max(joint_gradient_scale, ε), 0, 1)`，
      `joint_gradient_scale` 取该单元所有候选归一化 Sharpness_Score 的最大值，
      候选数为 1 时置 0
    - 低清晰度区域：先求该机位全部单元胜出 Sharpness_Score 的 10 百分位，
      所有候选都低于该分位的单元按连通域合并，记录世界坐标位置与面积，
      标识符 `fusion_low_sharpness_region`
    - _Requirements: 3.8, 3.9_

  - [x] 7.12 机位层降级路径与拒绝并入判据
    - 全部非锚点帧失败 → 降级为有效面积内中位 Sharpness_Score 最高的单张源图
      （同值取绝对路径字典序最小），ownership 全指向该源图，
      标识符 `fusion_single_frame_degraded`
    - 存在失败帧且成功帧 ≥ 2 → 仅用成功帧合成，逐个记录被排除源图路径与原因标识符
    - 空间支持 < 20% 或内点中位对称重投影误差 > 0.01×长边 → 拒绝并入任何机位，
      记录实测值与 `intra_station_rejected_from_group`
    - _Requirements: 12.1, 12.2, 12.5_

  - [x]* 7.13 属性测试：分组只由图像证据决定
    - **Property 1: 分组只由图像证据决定**
    - **Validates: Requirements 1.1**
    - 生成器 `arb_focus_bracket(plane)`

  - [x]* 7.14 属性测试：累计位移拆分位置唯一确定
    - **Property 3: 累计位移拆分位置唯一确定**
    - **Validates: Requirements 1.4, 1.5, 1.10**

  - [x]* 7.15 属性测试：不可用源图不影响其余分组
    - **Property 4: 不可用源图不影响其余分组**
    - **Validates: Requirements 1.6, 1.9**

  - [x]* 7.16 属性测试：锚点帧选择确定且变换恒等
    - **Property 5: 锚点帧选择确定且变换恒等**
    - **Validates: Requirements 2.1**

  - [x]* 7.17 属性测试：局部匹配接受条件的充要性
    - **Property 6: 局部匹配接受条件的充要性**
    - **Validates: Requirements 2.4, 2.5, 2.6**

  - [x]* 7.18 属性测试：局部配准不劣化则不回退
    - **Property 7: 局部配准不劣化则不回退**
    - **Validates: Requirements 2.7**
    - 生成器 `arb_intra_station_verdict_pair()`：一次抽三次抽到**已对齐**的层
      （`shift = (0, 0)`、无梯度），否则全局模型永远还带着整个位移，局部 refinement
      永远只会更好，需求 2.7 的比较只会有一个方向。实测 40 次抽样：10 次 Reverted
      （含 3 次两个中位数**恰好相等**）、9 次 Kept、21 次 Indeterminate
    - 断言 `Reverted ⟺ ¬(local < global)`、`Kept ⟺ local < global`（两侧样本数均 ≥ 6），
      回退后位移场与进入时的全局模型场逐位相等，Kept 时位移场确实被改动过
    - 非空洞性：把 `local_field_improves` 的 `<` 改成 `<=`（恰好相等的那些帧从
      Reverted 翻成 Kept）→ 本属性失败

  - [x]* 7.19 属性测试：配准覆盖率判定与定义一致
    - **Property 8: 配准覆盖率判定与定义一致**
    - **Validates: Requirements 2.8**
    - 覆盖率按需求原文用**两遍 spacing 的并集**独立重算并与生产报告比较；并集与最后一遍
      计数同时观测，实测 40 次抽样里 9 次两者不同，运行结束断言这种情形出现过
    - 需求 2.5 的邻域门用单遍网格这一区分在 `ControlPointGrid` 上单独断言
      （`absorb` 得并集，`neighbour_median` 只看被查询那一遍）
    - 非空洞性：把 `apply_native_pass` 的覆盖率分子换成 `pass.accepted_grid.accepted_count()`
      → 本属性失败；把 `INTRA_STATION_MIN_INLIER_AREA_COVERAGE` 改成 0.10 → 本属性失败

  - [x]* 7.20 属性测试：ownership 单元尺寸恒在规定范围
    - **Property 9: ownership 单元尺寸恒在规定范围**
    - **Validates: Requirements 3.1**
    - 单元边长 ∈ [8, 64] 对任意平面尺寸断言；512 单元下界断言为**充要**关系：
      长边 ≥ `8 × 511 + 1 = 4089` 原生像素。低于该分界点时需求 3.1 的两条子句互相矛盾
      （实测结论，见下方说明）；设计文档写的「64px 上限除外」实测为空例外
    - `floor` 而非 `ceil`：断言 `floor` 的单元数恒不少于 `ceil` 的，且断言运行中出现过
      `ceil` 会掉到 512 以下的平面
    - 非空洞性：把 `ownership_cell_size_px` 的 `floor` 改回 `ceil` → 本属性失败

  - [x]* 7.21 属性测试：清晰度比较使用相同采样
    - **Property 10: 清晰度比较使用相同采样**
    - **Validates: Requirements 3.2**
    - 用记录闭包把生产测量**实际索要的**每个采样坐标按顺序记下，逐元素比较位模式；
      断言每单元 5 个探针（中心 + 四象限对角）、窗口为 13 个整像素步长且没有别的整数步长
      比它更接近 32 原生像素
    - 接线在生产函数 `stitching::measure_focus_cells` 上断言：同一张图放在相差**恰好一个
      单元**的两个位置测量，结果必须逐位错位一格相等
    - 本任务在 `measure_focus_cells` 入口补上了任务 7.6 要求的那条显式断言
      （`plan.assert_matches(&geometry.sampling_plan())`），使「同一采样」成为被强制的性质
      而不只是约定；属性用 `catch_unwind` 断言外来采样方案会被拒绝
    - 非空洞性：删掉该断言 → 属性失败；`sharpness_sample_step_px` 的 `round` 改 `ceil`
      （窗口 39px）→ 属性失败；`measure_focus_cells` 里丢掉图层原点 `left/top` → 属性失败

  - [x]* 7.22 属性测试：不一致性取值与惩罚触发正确
    - **Property 11: 不一致性取值与惩罚触发正确**
    - **Validates: Requirements 3.3**
    - 用生产的低通差值序列独立重算中位数并与 `cell_disagreement` 比较（旧
      `mosaic::ownership_disagreement` 的高分位数语义会被这条断言抓住）；两侧被索要的
      单元内偏移同样用记录闭包逐元素比较
    - 惩罚断言充要性（不一致性用 `k / 40` 抽样，一次抽四次恰好落在 0.2 边界上，
      区分「超过 0.2」与「不低于 0.2」），并断言其后果：被惩罚的候选在数据项上永远赢不了
      当前底图
    - 非空洞性：把 `cell_disagreement` 的中位数改成 0.9 分位数 → 属性失败；
      把 `OWNERSHIP_DISAGREEMENT_VETO` 改成 0.25 → 属性失败

  - 说明（7.20 的实测结论，需求 3.1 的两条子句在小平面上不相容）：单元边长下界是 8，
    最后一列/行允许越过平面边界但仍然要有 owner，所以长边上放得下 512 个单元的充要条件是
    长边 ≥ 4089 个原生像素。低于该值时「边长 ∈ [8, 64]」与「长边不少于 512 个单元」
    不可能同时成立，生产实现保留前者（单元边长恒为 8），属性按充要关系断言并把分界点钉住。
    真实机位合成平面（9504px 长边）远在分界点之上，该不相容区间不可达。

  - [x]* 7.23 属性测试：多标签图割给出唯一最小代价 owner
    - **Property 12: 多标签图割给出唯一最小代价 owner**
    - **Validates: Requirements 3.4**
    - 生成器 `arb_cost_grid(w, h, labels)`，≤3×3、≤3 标签与穷举比较
    - 多标签 Potts 的 alpha-expansion 不保证精确全局最优；断言生产总代价不超过穷举最优值的
      `1.01` 倍，同时保持每个已覆盖单元唯一 owner、重复输入标签逐位一致与 tie-break 确定性
    - 执行记录（复核）：随机种子 `cc 0639d64c…` 找到反例（2×3、3 标签，6.62 > 1.01×6.53）。已修正 alpha-expansion 移动能量的重参数化（`seam_cut::cut_grid_weighted` 显式边权）并加入确定性标签顺序重启；种子写入 proptest-regressions，新增 `graph_cut_saved_counterexample_stays_within_property_12_bound`。两组真实门禁三站均为 graph_cut，输出字节不变。

  - [x]* 7.24 属性测试：低清晰度区域判定与分位数一致
    - **Property 14: 低清晰度区域判定与分位数一致**
    - **Validates: Requirements 3.8**

  - [x]* 7.25 属性测试：Sharpness_Confidence 值域与单调性
    - **Property 15: Sharpness_Confidence 值域与单调性**
    - **Validates: Requirements 3.9**

  - [x]* 7.26 属性测试：单帧机位直通
    - **Property 16: 单帧机位直通**
    - **Validates: Requirements 3.10**

  - [x]* 7.27 属性测试：输出像素来自单一真实照片
    - **Property 17: 输出像素来自单一真实照片**
    - **Validates: Requirements 3.6**

  - [x]* 7.28 属性测试：全部非锚点帧失败时的单帧降级
    - **Property 75: 全部非锚点帧失败时的单帧降级**
    - **Validates: Requirements 12.1**
    - 执行记录：种子 `permutation_seed=17155767282009907368,
texture_seed=9300639506941516608` 在 13d0af6b 与 c04687cf 隔离重放均通过；并发失败
      是全局 run ledger 交错，已由 run scope 修复。

  - [x]* 7.29 属性测试：部分帧失败时只用成功帧
    - **Property 76: 部分帧失败时只用成功帧**
    - **Validates: Requirements 12.2**
    - 执行记录：种子 `permutation_seed=9501257891083706328` 在 13d0af6b 与 c04687cf
      隔离重放均通过；已用同一 run scope 修复并发状态交错。

  - [x]* 7.30 属性测试：并入机位的拒绝条件
    - **Property 78: 并入机位的拒绝条件**
    - **Validates: Requirements 12.5**

  - [x]* 7.31 单元测试：输入与成员数边界、分析预算/控制点约束、图割超时注入
    - 输入数量 0/1/2/500/501（需求 1.8）
    - 机位成员数 1/2/47/48/49（需求 1.5）
    - 实际分析长边配置 = 1400、且 ≤2048，并断言 Stack_Report 记录值与该实际配置相等（需求 2.2）
    - 控制点原生实测间距满足 `s ≤ min(112px, 8c)`；覆盖 `L = 7088/8256/9504`、
      `c = 13/16/18` 时约 81.0/94.4/108.6px 的生产公式，并保持窗口 ≥32、搜索可达范围 ≤64
      （需求 2.3）
    - 用确定性可注入时钟触发图割超时，断言逐单元最低代价回退、`PerCellFallback` 状态、
      `fusion_graph_cut_timeout` ledger 标识和 Stack_Report 状态/耗时字段（需求 3.5）
    - 用窄临时变异分别破坏输入上限、成员上限、分析报告值、间距上限与图割超时回退状态，证明对应测试
      会失败后逐项恢复
    - _Requirements: 1.5, 1.8, 2.2, 2.3, 3.5_

- [x] 8. 阶段 3 检查点
  - 验证合成输出的每个已覆盖像素逐位等于其 owner 源像素（Property 17）；
    确保 `mosaic.rs` 的 24 个单元测试与 `seam_cut::cut_grid` 的穷举对齐测试全部通过，
    遇到问题询问用户。

- [x] 9. 阶段 4：拼接层几何（拓扑 + 位姿 + 闭环）

  - [x] 9.1 实现 `topology.rs` 的行列索引推断
    - 仅依据机位间估计中心位移：水平位移 > 单机位有效宽度 0.5 倍判为不同列；
      先按水平位移一维聚类得列，再在列内按垂直位移排序得行；
      列索引按世界 x 降序、行索引按世界 y 升序（从上到下、从右到左的蛇形）
    - 文件名仅在估计位移完全相同时 tie-break
    - 行列索引无法唯一确定的机位标记 `topology_index_ambiguous`，
      与其余全部机位在预算内配对
    - `StationTopology` 类型中不含任何 `Matrix3`（需求 5.4）
    - _Requirements: 5.1, 5.2, 5.4, 5.7_

  - [x] 9.2 实现 Candidate_Adjacency 分类、候选分数与搜索预算
    - 三类候选：同列相邻 ≤2、同行相邻 ≤2、跨列相邻 ≤4
    - 候选分数复用 `panorama_transform_overlap_support()`
      （估计重叠面积占单个机位有效面积的比例）
    - 预算沿用 `FOCUS_AUTO_ORDER_EXHAUSTIVE_MAX_SOURCES = 64` 的「小规模穷举」思路，
      但改为**机位级**：机位数 ≤64 时输出全部机位对且不截断；
      否则预算 = `8 × station_count`，按分数降序保留，记录
      `truncated_count` 与 `truncation_min_score`，标识符 `topology_candidates_truncated`
    - 把行列索引与每个候选的机位对、邻接类型、候选分数写入 `topology`
    - _Requirements: 5.3, 5.5, 5.6, 5.9_

  - [x] 9.3 把机位间证据限制为 Virtual_Tile 覆盖像素与 Consensus_Feature
    - 主路径：在 Virtual_Tile 的 `coverage` 已覆盖像素上直接提特征做机位间匹配
    - 补充路径：源图级证据只有 `independent_support ≥ 2` 才计入内点
    - **删除** `FOCUS_GROUP_SINGLE_EDGE_WEIGHT_FACTOR = 0.02`
      （`panorama_stitching.rs:95`）对单源未确认证据的降权保留逻辑
      （调用点见 `panorama_stitching.rs:10160/10394/10945`），改为丢弃并计数上报
      `station_relation_single_layer_evidence_discarded`
    - _Requirements: 6.1, 6.2_

  - [x] 9.4 收紧 Station_Relation 的接受门槛到世界像素绝对量
    - 新增 `STATION_RELATION_MIN_INLIERS = 24`（替代 `FOCUS_MODEL_MIN_INLIERS = 8`
      在机位关系上的使用）
    - 新增 `STATION_RELATION_MAX_MEDIAN_ERROR_PX = 3.0`，替代
      `FOCUS_GROUP_CONSENSUS_MAX_ERROR_RATIO = 0.006`（`panorama_stitching.rs:94`，
      在 9504px 长边上 ≈57 世界像素）作为接受判据；
      现有比率型用法（`panorama_stitching.rs:10165/10390/10422/10913`）逐处改为绝对像素量
    - 新增尺度比 `[0.95, 1.05]`（复用 7.1 的奇异值几何均值函数）
    - 内点空间支持改用需求 6.4 的定义（内点凸包面积 / 重叠面积）≥ 0.20，
      替代 `FOCUS_PROJECTIVE_MIN_SPATIAL_SUPPORT = 0.22`
    - 用 `focus_overlap_quality()` 的返回值补三项复核：低频亮度均值相对差 ≤20%、
      边缘强度比 ∈ `[0.7, 1.4]`、边缘方向中位差 ≤10°，
      替代现有 `edge_ncc ≥ −0.10` / `edge_orientation ≥ −0.10`
    - 保留 `homography_preserves_focus_orientation()` 的凸四边形检查
    - 每个被拒绝候选记录对应 `station_relation_*` 标识符，
      按标识符汇总计数写入 `station_relations.rejected_by_reason`
    - 同步更新 `tests/focus-stack-quality-contract.mjs`：新增
      `STATION_RELATION_MIN_INLIERS === 24`、
      `STATION_RELATION_MAX_MEDIAN_ERROR_PX === 3.0` 断言
    - _Requirements: 6.3, 6.4, 6.5, 6.8, 6.9, 6.10_

  - [x] 9.5 Local_Scale 比值约束与不连通拒绝
    - 对每个机位位姿 `H` 在瓦片有效区域采样网格点，计算雅可比行列式绝对值平方根，
      要求 `max / min ≤ 1.10`；超限拒绝该候选位姿并回退生成树初值，
      标识符 `station_pose_local_scale_exceeded`；不把任何自由度强制固定为恒等
    - 不连通分量：把 `solve_focus_capture_group_poses()` 现有的
      `return locked_homographies.clone()` 静默回退改为**拒绝输出完整拼接结果**，
      列出每个分量的成员绝对路径与成员数量，标识符 `geometry_disconnected`（Rejected），
      错误提示包含分量数量与「按连续场景重新分组」「补拍增加重叠」两类动作
    - _Requirements: 6.6, 6.7, 12.4_

  - [x] 9.6 删除 `refine_focus_group_projective_tree_poses()` 的约束预剔除
    - 删除该函数中 `if dx.hypot(dy) > maximum_constraint { continue; }` 的预剔除
      （违反需求 7.2），大残差边改由 M-estimator 权重压制，
      并把其最终权重写入 Stack_Report
    - 保留 `optimize_focus_group_projective_poses()` 中 `if points.len() < 6 { continue; }`
      （数值必需）
    - 使参与联合优化的约束数量等于被接受的水平、垂直、跨列关系总数，
      写入 `closure.participating_constraints`
    - _Requirements: 7.2_

  - [x] 9.7 生成树边权与 M-estimator 权重函数
    - 生成树边权从综合 `score` 改为内点数量，相同内点数按行列索引升序 tie-break；
      生成树解仅作初值
    - 把现有 `robust_limit / magnitude` 形状（`panorama_stitching.rs:10422`）替换为分段线性权重：
      `w(r) = 1.0 (r ≤ 2.0)`；`1.0 − 0.9·(r − 2.0)/4.0 (2.0 < r ≤ 6.0)`；
      `0.1·(6.0/r) (r > 6.0)`，单调不增、连续，满足 `w(2.0) ≥ 0.9`、`w(6.0) ≤ 0.1`
    - _Requirements: 7.1, 7.3_

  - [x] 9.8 角点修正上限与收敛判据
    - 把 `FOCUS_GROUP_MAX_CENTER_CORRECTION_RATIO = 0.50`（`panorama_stitching.rs:96`，
      在 9504px 长边上 ≈4752px，用于 `panorama_stitching.rs:10484/11422`）替换为
      `CLOSURE_MAX_CORNER_CORRECTION_PX = 256` 与
      `limit = min(0.10 × 该机位重叠区较短边, 256.0)`，按**四个角点位移**判定，
      超限按上限截断（不丢弃整解），记录实际最大角点位移与被截断机位数，
      标识符 `closure_correction_clamped`
    - 迭代从固定 8 次 + 0.01px 变化判据改为：鲁棒加权总残差相邻两次相对下降 < 1e-4 收敛，
      迭代上限 100
    - 记录 `participating_constraints`、`iterations`、`residual_median_px`、`residual_p95_px`
    - 同步更新 `tests/focus-stack-quality-contract.mjs`：新增
      `CLOSURE_RESIDUAL_MEDIAN_LIMIT_PX === 2.0`、
      `CLOSURE_MAX_CORNER_CORRECTION_PX === 256` 断言
    - _Requirements: 7.4, 7.5_

  - [x] 9.9 闭环验收、直接相连对复核与回退
    - 中位残差 > 2.0 世界像素或 100 次迭代未收敛 → 保留生成树初值，
      标识符 `closure_unreliable_residual` / `closure_unreliable_iterations`
    - 通过后复核每对直接相连机位重叠区重投影误差 P95 ≤ 3.0 世界像素，
      记录所有直接相连对中该 P95 最大值 `max_direct_pair_p95_px`；
      任一超限 → 保留生成树初值，标识符 `closure_unreliable_pair_p95` 并列出该对行列索引
    - 被接受关系数 ≤ 机位数 − 1 → 跳过联合优化直接输出生成树解，
      标识符 `closure_no_constraints`
    - 闭环降级时几何置信度标记降级但 Quality_Gate 全部判据仍然生效
    - _Requirements: 7.6, 7.7, 7.8, 7.9, 7.10, 12.3_

  - [x] 9.10 把合成顺序 tie-break 切换为行列索引
    - 把阶段 1 中改为世界坐标排序的
      `focus_source_order_group_compositor_order()` tie-break 切换为行列索引
    - _Requirements: 14.6_

  - [x] 9.11 实现「拒绝优先于降级」的统一决策
    - `Degradation_Manager` 收集本次运行全部生效路径；存在任一 `Rejected` ⇒ 结果为拒绝输出
    - 拒绝时最终结果文件走 `.tmp` + rename，拒绝时删除 `.tmp`，保证输出目录无残留；
      Stack_Report 与诊断预览始终保留
    - _Requirements: 12.9, 12.10_

  - [x]* 9.12 属性测试：行列索引单射且由位移决定
    - **Property 25: 行列索引单射且由位移决定**
    - **Validates: Requirements 5.1, 5.2**
    - 生成器 `arb_scan_grid(plane)`

  - [x]* 9.13 属性测试：候选邻接数量与预算约束
    - **Property 26: 候选邻接数量与预算约束**
    - **Validates: Requirements 5.3, 5.5, 5.6**

  - [x]* 9.14 属性测试：索引歧义机位与全部机位配对
    - **Property 27: 索引歧义机位与全部机位配对**
    - **Validates: Requirements 5.7**

  - [x]* 9.15 属性测试：机位间证据仅取自已覆盖像素
    - **Property 28: 机位间证据仅取自已覆盖像素**
    - **Validates: Requirements 6.1, 6.2**

  - [x]* 9.16 属性测试：Station_Relation 接受条件的充要性
    - **Property 29: Station_Relation 接受条件的充要性**
    - **Validates: Requirements 6.3, 6.4, 6.8**

  - [x]* 9.17 属性测试：拒绝原因标识符与缺陷一一对应
    - **Property 30: 拒绝原因标识符与缺陷一一对应**
    - **Validates: Requirements 6.5, 6.9, 6.10**

  - [x]* 9.18 属性测试：位姿保留 8 自由度且局部尺度有界
    - **Property 31: 位姿保留 8 自由度且局部尺度有界**
    - **Validates: Requirements 6.6**

  - [x]* 9.19 属性测试：不连通机位图拒绝输出
    - **Property 32: 不连通机位图拒绝输出**
    - **Validates: Requirements 6.7, 12.4**

  - [x]* 9.20 属性测试：生成树最大支持且 tie-break 稳定
    - **Property 33: 生成树最大支持且 tie-break 稳定**
    - **Validates: Requirements 7.1**

  - [x]* 9.21 属性测试：所有被接受关系参与联合优化
    - **Property 34: 所有被接受关系参与联合优化**
    - **Validates: Requirements 7.2**

  - [x]* 9.22 属性测试：鲁棒权重单调不增且满足两端边界
    - **Property 35: 鲁棒权重单调不增且满足两端边界**
    - **Validates: Requirements 7.3**

  - [x]* 9.23 属性测试：位姿修正被上限截断
    - **Property 36: 位姿修正被上限截断**
    - **Validates: Requirements 7.4**
    - 执行记录：种子 `787596336838008496`（`use_absolute_cap=true`）在 13d0af6b 与
      c04687cf 隔离重放均通过；并发失败来自全局 closure sink 交错，已由 run scope 修复。

  - [x]* 9.24 属性测试：联合优化终止条件
    - **Property 37: 联合优化终止条件**
    - **Validates: Requirements 7.5**

  - [x]* 9.25 属性测试：被接受的闭环解满足残差上界
    - **Property 38: 被接受的闭环解满足残差上界**
    - **Validates: Requirements 7.6, 7.8**

  - [x]* 9.26 属性测试：闭环不可靠时位姿不被改写
    - **Property 39: 闭环不可靠时位姿不被改写**
    - **Validates: Requirements 7.7, 7.9**

  - [x]* 9.27 属性测试：无闭环约束时直接输出生成树解
    - **Property 40: 无闭环约束时直接输出生成树解**
    - **Validates: Requirements 7.10**

  - [ ]* 9.28 属性测试：闭环降级不放松画质判据
    - **Property 77: 闭环降级不放松画质判据**
    - **Validates: Requirements 12.3**

  - [x]* 9.29 属性测试：拒绝优先于降级
    - **Property 80: 拒绝优先于降级**
    - **Validates: Requirements 12.9**

  - [x]* 9.30 属性测试：路径选择由机位数量决定并被记录
    - **Property 92: 路径选择由机位数量决定并被记录**
    - **Validates: Requirements 15.1, 15.11**

  - [x]* 9.31 单元测试：`StationTopology` 不含位姿参数
    - 断言 `StationTopology` / `CandidateAdjacency` 类型中不含 `Matrix3` 字段（需求 5.4）
    - _Requirements: 5.4_

- [x] 10. 阶段 4 检查点
  - 验证合成扫描网格上闭环中位残差 ≤2.0 世界像素、所有直接相连对 P95 ≤3.0 世界像素；
    确保所有测试通过，遇到问题询问用户。
  - 执行记录（`langyuan-10` 真实门禁根因，用落盘的真实 Virtual_Tile 与 NEF 独立测得，
    OpenCV SIFT + RANSAC；临时诊断代码已撤回）：
    - 粗先验本身错误：源图级组件恢复接受了伪 coarse bridge `DSC_3680 ↔ DSC_3688`
      （11 个一致层，但 SIFT 证实两机位无重叠），把机位 `3687–3689` 放到错误位置
      （瓦片 1→2 先验残差 dy ≈ +3382px、dx ≈ +417px；源图级尺度 0.94，按两级相邻放大率推算应约 0.81）；
      真实邻接是 `3684–3686 ↔ 3687–3689`（瓦片 SIFT 685 内点，NEF `3684→3687` 803 内点）。
      `0↔2` 先验只中度偏差（约 1.2° 旋转，内点处 dy p10/p90 = −27/+10px）。
    - 原生分辨率稠密探针在融合瓦片上不可用：即使用 SIFT 单应性（中位误差约 1.6px）作种子，
      13px 与 33px 窗口都只得到 58–92 个双向匹配，前向峰值的结构方向差中位数仍为 32–35°
      （即约 37° 是伪峰特征，不是雅可比/旋转错误），支持度 < 0.20。同一探针在 1/4、1/8 面积平均
      金字塔层上有效：`0→2` 仅凭粗先验即被接受（1/8：404 双向、257 平移一致、支持 0.41；
      1/4：563 / 248 / 0.26）。
    - 无先验粗关系可行：分析分辨率（1/3–1/4）覆盖像素上的多尺度 FAST/BRIEF + RANSAC 找到
      `1→2`（253–260 内点，与 SIFT 中位差 7–10px）、`0→2`（63–123 内点，精度较差），`0→1` 无解。
    - 但门禁冲突仍在：`1→2` 在世界系瓦片上的尺度比 ≈0.85–0.87 且透视显著（瓦片 1 由错误的源图级位姿
      渲染），需求 6.3 的 `[0.95, 1.05]` 会拒绝它；NEF 实测相邻机位重叠区局部放大率为
      1.096–1.119（相似变换残差 13–20px、单应性 1.2–1.6px），若按锚点帧渲染瓦片，
      需求 6.3 / 6.6 同样不成立。仅调 Virtual_Tile 匹配器无法通过该门禁。
    - 用户选择 A（需求阈值 `STATION_RELATION_*` 不变）。实现：
      - Coverage_Mask 面积平均金字塔 `[16, 8, 4]`（小瓦片只保留一个原生层）；粗层跑有界双向探针
        （层内容差 1.5px、取内点最多的模型、修正量受证据区域限界、边缘方向在 σ=1 结构平面上测）逐层做种子；
        最细层按模型形变做 15×15 NCC 精修，RANSAC + LO，对称容差 √2×3.0px，网格单元配额保留内点凸包；
      - 无先验 FAST/BRIEF 覆盖特征提议只在先验一致的簇之间发起，只作种子、从不成为权威关系；
        `plan_virtual_tile_prior_repairs`（DSU）按已验证关系修复与先验矛盾机位的源图位姿，被修复机位重渲染一次；
      - 求解器 6.8 复核在最细层之上两个 octave 的 σ=1 平面上测，只采样两瓦片 ±4px 邻域均被覆盖处
        （未覆盖像素曾把 `1↔2` 边缘强度比抬到 1.3994、低频差 0.129；只看覆盖处约 1.01 / 0.002）；
      - Virtual_Tile 阶段替换了源图级位姿（修复或权威求解）后，全画布取合成画布 / render_scale；
      - Stack_Report 新增 `pyramid_factors` / `measurement_factor` / `seed` / `prior_free_inliers` /
        `after_prior_repair` 与 `prior_repairs`；`intra_station` 按路径替换重渲染机位的帧记录。
    - run1 失败：无先验 `1→2` 提议被粗层方向门拒（原始 1/8 平面 15.54° > 15°）→ 粗层改在 σ=1 结构平面上测。
    - run2 失败：修复后 `full_canvas_height ≥ rendered_height` 不成立 → 修复后全画布取合成画布。
    - 最终 `langyuan-10` 通过：8788x10949，全画布同尺寸，preview / 输出已写出且与 run3 逐字节一致；
      - `0↔2`：`rendering_prior`，1/4 层，754 内点，中位 1.641px，尺度 1.0137，凸包支持 0.2007；
      - 修复：机位 1 经无先验 `1→2`（153 特征内点 → 1157 验证内点）接到机位 2，中心移 3605.5px、尺度 0.7856；
      - `1↔2`（修复后重渲染）：`rendering_prior`，1/4 层，1087 内点，中位 2.127px，尺度 1.0016，凸包支持 0.2160；
      - `0↔1` 无重叠被拒；闭环 `no_constraints`（2 条生成树边），1 个连通分量；机位内帧无重复 Source_RAW。
    - `wenyuan-10`：首跑 `full_canvas_width ≥ rendered_width` 失败（无修复，权威求解本身把合成画布拓到 8816 列，
      全画布仍取源图级画布）→ 改为上面的「替换位姿后取合成画布」；重跑通过：8816x8666，全画布同尺寸，
      preview / 输出已写出，无修复，1 个连通分量，机位内帧无重复 Source_RAW；
      - `0↔1` 5977 内点 / 0.988px / 尺度 1.0000 / 支持 0.8544，`0↔2` 813 / 2.171px / 1.0188 / 0.3115；
      - `1↔2` 曾被求解器误拒（中位误差 8329px、低频差 0.406、方向差 44.2°）：Virtual_Tile 的
        `canonical_homography` 按 `(left, right)` 键方向存储，内容方向为 2→1 时求解器交换了点对却未求逆
        → Virtual_Tile 关系不再填 `canonical_homography`（`None`），求解器按内容方向至多求逆一次；
        修复后 `1↔2` 接受（681 内点 / 2.195px / 内容方向 2→1 尺度 0.9840 / 支持 0.2322），3 条关系、0 拒绝；
      - 闭环真正参与求解（3 条约束、2 次迭代，残差中位 0.74px、P95 0.78px），但判为 `unreliable`
        （`closure_unreliable_pair_p95`：`1↔2` 直接对 P95 33.3px > 3.0，取自重叠矩形 11×11 采样格）
        → 回退生成树位姿，输出与修复前逐字节一致。
      - 该 P95 是真实的环路不一致，不是未覆盖区外推：只取两瓦片都覆盖的采样点仍为 27.0px
        （101×101 密格 30.2px）；`1↔2` 自身内点在生成树位姿下中位 2.76px、P95 9.53px（关系自拟合 P95 2.75px）；
        不一致近似一个旋转（瓦片 1 中 dy 随 x 变化 −0.008，内点带 2500px 上约 20px，去掉仿射项后残差 P95 2.74px）；
        1/16 质量平面上的独立 SIFT 关系同样闭不上环（P95 8.1px）。仅平移的闭环修正无法吸收该旋转，
        按需求 7.9 回退生成树是正确结论；全局单应性在该重叠区 P95 > 3.0，正是需求 8.1 局部形变（阶段 5）的启用条件。
    - 需求 6.4 重叠面积改为覆盖重叠（用户确认）：`langyuan-10` `0↔2` 凸包支持 0.2007 曾只高出门槛 0.0007，
      而矩形重叠里 8% / 13% 是未覆盖角——没有内容、不可能有内点、也不会由两瓦片合成。现在 Virtual_Tile 证据的
      重叠面积只计两瓦片 Coverage_Mask 都覆盖的像素（`StationOverlapMeasure::Covered`，求解器在最细匹配层计数），
      匹配器预拟合 / 残差各阶段 / 拟合后 / 打磨收尾与求解器复核同一口径，源图证据仍取矩形交；design.md 已注明。
      两套素材接受的关系、内点、位姿、闭环不变，输出与改前逐字节一致；支持度 langyuan `0↔2` 0.2007 → 0.2160、
      `1↔2` 0.2160 → 0.2462，wenyuan `0↔1` 0.8544 → 0.8644、`0↔2` 0.3115 → 0.3518、`1↔2` 0.2322 → 0.2585。
    - 验证：新增 `panorama_virtual_tile_tests.rs` 30 个回归测试（金字塔因子、面积平均平面与映射、
      质量平面与其覆盖、覆盖感知 6.8（直接测量与经求解器）、覆盖重叠面积与 6.4 支持（直接与经求解器）、
      修复规划、有界修正、残差模型选择、打磨拟合与打磨匹配恢复、帧记录按路径替换、报告字段默认值、
      二维扫描网格闭环）与 `alignment_tests` 中 3 个（结构平面只改方向测量、键方向关系交给求解器、两种内容方向都接受）；
      3×3 蛇形扫描网格上生成树漂移 3.74px，闭环后中位残差 0.52px、所有直接相连对 P95 ≤ 0.93px（3 次迭代）；
      `cargo test --lib` 558 通过 / 27 忽略；clippy 在新代码上无新告警；两个 Node 契约通过。

- [ ] 11. 阶段 5：局部形变与色调

  - [x] 11.1 实现 `residual_warp.rs` 的启用条件与网格
    - `regions` 默认为空（恒等映射）；仅当某重叠区全局单应性重投影误差 P95 > 3.0 世界像素
      时为该区创建 `WarpRegion`
    - 新增常量 `RESIDUAL_WARP_NODE_STEP_PX = 64`，每方向 ≥4 节点；
      复用 `mosaic.rs` 的 `Field<2>` 容器承载位移场
    - 记录该区世界坐标范围与启用前实测 P95
    - _Requirements: 8.1, 8.2_

  - [x] 11.2 实现节点位移三项约束与外推、衰减、单元回退
    - 新增 `RESIDUAL_WARP_MAX_NODE_DISPLACEMENT_PX = 32`、
      `RESIDUAL_WARP_MAX_NEIGHBOUR_DELTA_PX = 8`、`RESIDUAL_WARP_BOUNDARY_DECAY_PX = 128`
    - 往返误差 ≤1.0 世界像素（复用 `refine_warped_patch()` 的双向匹配得反向位移），
      三项全部满足才写入节点位移
    - 无效节点用距离 ≤3 个节点的有效邻域按 `NATIVE_FIELD_RADIUS` 高斯加权外推，
      范围内无有效邻域则置 0
    - 距启用区边界 128 世界像素内位移乘 `smoothstep(d / 128)` 单调衰减至 0，
      区域外严格保持全局单应性
    - 逐网格单元比较 P95，不优于全局单应性的单元位移清零并记录该单元世界坐标范围，
      标识符 `residual_warp_cell_reverted`
    - 往返校验通过的匹配点 < 16 → 该区保持恒等，标识符 `residual_warp_insufficient_evidence`
    - 同步更新 `tests/focus-stack-quality-contract.mjs`：新增四个 `RESIDUAL_WARP_*` 常量断言
    - _Requirements: 8.3, 8.4, 8.5, 8.6, 8.8, 8.9, 8.10_

  - [~] 11.3 把 `ResidualWarp` 与 `tile_to_world` 复合成单次采样
    - `Tile_Compositor` 对每个目标像素求
      `tile_coord = warp_inverse(tile_to_world_inverse(world_coord))` 后一次性采样，
      使瓦片 → 世界重采样不超过 1 次
    - 执行记录：在 Wenyuan 实验中启用站位双向匹配 P95=9.5265px 的区域，输出保持基线
      SHA256，但没有记录到启用区域的 post-warp P95 下降，故未接入生产路径；实验补丁保存在
      `/private/tmp/raw-editor-gate/residual-11.3.patch`（仓库外），两组最终门禁均为 `residual_warp.identity=true`、空区域。
    - _Requirements: 8.7_

  - [x] 11.4 调整 `PhotometricOptions` 参数并补齐偏移项
    - `photometric.rs`：`min_sample_value` 0.01 → 0.02、`max_sample_value`
      `1 − 1/65535` → 0.98、`min_samples_per_pair` 32 → 1024、
      `max_abs_log_gain` `ln 2` → `ln 1.25`、`allow_linear` 保持 false
    - 把 `mosaic.rs:41` 的 `STREAMING_GROUP_GAIN_MAX_LOG = 0.45`（≈1.57×）收紧到 `ln 1.25`，
      同步检查 `mosaic.rs:409/483/557` 三处 `clamp` 调用点
    - 扩展模型为 `out = gain · in + offset`：先用现有 log 增益求解得 `gain`，
      再对残差求稳健中位数得 `offset`，`|offset| ≤ 0.02` 归一化满量程
    - 增益/偏移超范围时截断到最近边界，记录求解值与截断值，标识符 `tone_gain_clamped`
    - 同步更新 `tests/focus-stack-quality-contract.mjs`：新增 `PhotometricOptions`
      默认值断言（0.02 / 0.98 / 1024 / `ln 1.25`）
    - _Requirements: 9.3, 9.9_

  - [x] 11.5 实现 `tone.rs` 的低频带、MAD 过滤与高频残差保持
    - 低通 σ ≥ 64 世界像素，作用在亮度与 R、G、B 分量；
      实现上在 1/16 降采样网格上求低频场（σ = 4 网格像素）后双线性放大，
      与现有 `tone_field()` 的 `Field<3>` 机制一致
    - 把现有 `max_pair_log_scatter` 散度门改为显式「排除与样本中位数偏差 > 3×MAD 的样本」，
      过滤后重新检查 `min_samples_per_pair = 1024`，不足时该对用恒等增益零偏移，
      标识符 `tone_insufficient_samples`
    - 输出 = `low_corrected + (owner_source − owner_source_low)`，
      使高频残差逐像素等于 owner 源图
    - 无有效样本区域用恒等增益零偏移
    - _Requirements: 9.1, 9.2, 9.4, 9.7, 9.10_

  - [x] 11.6 把色调阶段顺序移到接缝之后并复核边界带色差
    - 现有 `progressive_seam_stitcher` 是「先估曝光、再选接缝」，改为
      在 `Tile_Compositor` 完成接缝与输出 Ownership_Map **之后**调用 `Tone_Harmonizer`；
      接缝选择不得读取任何已校正像素
    - 校正后对 Ownership_Map 做逐像素相等断言
    - 相邻 Owner_Region 边界两侧各 16 世界像素带内计算低频均值 Delta_E00，
      > 1.5 时记录该对、实测值与边界世界坐标，标识符 `tone_boundary_delta_e_exceeded`，
      > 色调协调状态标记降级
    - _Requirements: 9.5, 9.6, 9.8, 9.11_

  - [x] 11.7 删除三个色调环境变量开关
    - 删除 `RAW_EDITOR_ENABLE_GLOBAL_PHOTOMETRIC`（`Tone_Harmonizer` 无条件调用）、
      `RAW_EDITOR_ENABLE_SPATIAL_TILE_EXPOSURE_GAIN`（空间项代码保留但不接线，
      由 `allow_linear = false` 单点控制）、`RAW_EDITOR_SKIP_RGB_TILE_EXPOSURE_GAIN`
    - 同步更新 `tests/focus-stack-quality-contract.mjs`：新增「`Tone_Harmonizer`
      入口函数体内不出现 `std::env::var`」断言
    - _Requirements: 15.2_

  - [x] 11.8 属性测试：局部形变的启用范围精确
    - **Property 41: 局部形变的启用范围精确**
    - **Validates: Requirements 8.1, 8.2**
    - 执行记录：新增 P41（100 cases）：低 P95 不分配区域，高 P95 使用 64px 网格且边界与节点条件固定。

  - [x] 11.9 属性测试：写入的节点位移同时满足三项约束
    - **Property 42: 写入的节点位移同时满足三项约束**
    - **Validates: Requirements 8.3, 8.4, 8.5**
    - 执行记录：新增 P42（100 cases）：写入位移逐节点检查 32px、邻差 8px 与往返 1px 三项约束。

  - [x] 11.10 属性测试：局部形变不劣化则不回退
    - **Property 43: 局部形变不劣化则不回退**
    - **Validates: Requirements 8.6**
    - 执行记录：新增 P43（100 cases）：逐单元比较全局/残差 P95，验证不劣化单元回退。

  - [ ]* 11.11 属性测试：瓦片到世界只重采样一次
    - **Property 44: 瓦片到世界只重采样一次**
    - **Validates: Requirements 8.7**

  - [x] 11.12 属性测试：无效节点外推范围有界
    - **Property 45: 无效节点外推范围有界**
    - **Validates: Requirements 8.8**
    - 执行记录：新增 P45（100 cases）：无效节点只从 3 节点半径内有效邻域外推。

  - [x] 11.13 属性测试：形变位移在边界单调衰减为零
    - **Property 46: 形变位移在边界单调衰减为零**
    - **Validates: Requirements 8.9**
    - 执行记录：新增 P46（100 cases）：边界 128px 衰减单调至零且区域外恒等。

  - [x] 11.14 属性测试：形变证据不足则保持恒等
    - **Property 47: 形变证据不足则保持恒等**
    - **Validates: Requirements 8.10**
    - 执行记录：新增 P47（100 cases）：少于 16 个通过观测时区域保持恒等并标记证据不足。

  - [x]* 11.15 属性测试：色调样本过滤条件的充要性
    - **Property 48: 色调样本过滤条件的充要性**
    - **Validates: Requirements 9.1, 9.4**
    - 执行记录：P48 通过 `PairField` 夹具复核双覆盖、Coverage、亮度可用性与 3×MAD 一致性
      筛选；100 cases 通过。已接受关系的配对条件仍由 `harmonize_tone_tiles` 的关系列表控制。

  - [ ]* 11.16 属性测试：高频残差逐像素等于 owner
    - **Property 49: 高频残差逐像素等于 owner**
    - **Validates: Requirements 9.2**

  - [x]* 11.17 属性测试：色调增益与偏移恒在范围内
    - **Property 50: 色调增益与偏移恒在范围内**
    - **Validates: Requirements 9.3, 9.7, 9.9, 9.10**
    - 执行记录：P50 以 100 cases 覆盖求解值、截断后的应用值、记录字段，以及少于 1024 个样本时的恒等回退，全部通过。

  - [ ]* 11.18 属性测试：色调不改变接缝与 ownership
    - **Property 51: 色调不改变接缝与 ownership**
    - **Validates: Requirements 9.5, 9.6**

  - [x]* 11.19 属性测试：Owner_Region 边界低频色差有界或被记录
    - **Property 52: Owner_Region 边界低频色差有界或被记录**
    - **Validates: Requirements 9.8, 9.11**
    - 执行记录：P52 通过真实 `harmonize_tone_tiles` 生成边界测量；100 cases 对每个超出
      1.5 的结果检查 `tone_boundary_delta_e_exceeded` 降级记录，否则断言不超过阈值。

  - 执行记录：11.1–11.2、11.4–11.7 已实现并保留提交；残差与色调报告沿用 run sink，
    默认 Tone_Harmonizer 已在 Ownership_Map 后调用。11.3 的实际站位 P95/观测接线待粘贴到
    `panorama_stitching.rs`；11.8–11.19 可选属性测试暂未新增。

- [~] 12. 阶段 5 检查点
  - 验证任意曝光差的合成扫描网格上边界带 Delta_E00 ≤1.5，
    且输出高频残差逐像素等于 owner 源图高频残差；确保所有测试通过，遇到问题询问用户。

    - 执行记录：已落地残差场启用门槛、64px 网格、三项位移约束、外推与边界衰减；
      色调默认值收紧为 0.02/0.98/1024/ln(1.25)，增加有界 affine offset、3×MAD 过滤、
      1/16 低频场和 owner 高频残差保持；移除三个旧色调环境变量。两组真实门禁和完整测试已通过，
      但 langyuan-10 报告边界 Delta_E00 最大 30.635（阈值 1.5），且 residual warp 的实际
      站位 P95 接线仍待补入，故本检查点保持 [~]。
    - 执行记录（色调修复）：合并后的 Tone_Harmonizer 在真实数据上制造了整站亮度台阶
      （langyuan-10 底部站 ×0.88、黑边 1.7→6.0），原因依次为：所有相邻瓦片都当作色调关系
      （含被拒绝的 Station_Relation 0↔1）；逐对最小二乘仿射把斜率压到 0.83 并以 +0.057 偏移补偿，
      偏移截断到 0.02 后变成整站变暗；锚定第 0 站的生成树沿链传递误差；样本在 16384 处截断，
      只取重叠区顶部若干行；单侧低频场在覆盖边缘被内容梯度拉偏；应用时按 16px 最近邻取场。
      现改为：只取已接受 Station_Relation（站位求解经 `tone::record_run_topology` 发布），
      在两瓦片共同覆盖上构造成对低频场并取全部样本，3×MAD 作用于 log 比，关系取中位数 log 比，
      用 `photometric::solve_pairwise_differences` 联合求解（Huber、零均值，无锚点），
      偏移为残差中位数的联合解，应用时双线性放大；报告中的瓦片与对标识改为站位序号。
      新增合成 3×2 扫描网格属性（每站 RGB 曝光 [0.88, 1.12] 独立随机，三例）：状态 Applied、
      边界带 Delta_E00 ≤ 1.5、同一 owner 内校正量逐像素连续（高频残差不变）。
      真实门禁：langyuan-10 增益 0.981–1.011、偏移 ≈0、无截断，输出与阶段 5 之前相比逐区比值
      1–99% 分位 0.984–1.011，黑边 3.0/255 不变；边界窗口 46 个（恒等基线 49），集中在右侧
      反光墙面/玻璃杆与左侧瓦片暗角，max 16.85。wenyuan-10 三个关系成环，增益 0.969–1.024，
      比值 0.974–1.016，边界窗口 76 个（恒等基线 171），max 19.19 位于画幅右缘蓝灰墙面的
      真实照度差。两处剩余超限都是内容或局部照度差，单站全局增益无法也不应抹平，按需求 9.11
      记录并标记降级。residual warp 站位 P95 接线（11.3）仍待补入，本检查点保持 [~]。

- [ ] 13. 阶段 6：Tile_Compositor 成为默认路径

  - [x] 13.1 新增 `StackCompositorChoice` 设置项
    - 提前于阶段 3–5 执行，原因见执行记录
    - 枚举 `LayeredVirtualTile`（默认）/ `ProgressiveSeamTile` / `StreamingMosaic` /
      `LegacySingleLayerMosaic`，由设置项而非环境变量控制，对照开关默认关闭
    - 把所选路径的稳定标识符写入 `report.selected_path`：
      `layered_virtual_tile` / `progressive_seam_tile` / `streaming_mosaic` /
      `legacy_single_layer_mosaic` / `single_station`
    - 机位数 < 2 时仅执行机位层合成、不执行机位间位姿求解，标识符 `single_station`
    - _Requirements: 15.1, 15.2, 15.3, 15.10, 15.11_

  - [~] 13.2 接缝代价补齐「到有效覆盖边界的距离」项
    - 在 `compositor.rs` 复用 `find_adaptive_seam()` 的最小代价搜索，
      代价改为 `α · overlap_disagreement + β · boundary_penalty`，
      对距自身有效覆盖边界 < 16 世界像素的候选像素施加 ≥1.0 惩罚
      （与 ownership 数据项同量纲）
    - 只在两侧 `coverage` 都已覆盖的像素上搜索接缝
    - 有效重叠宽度 < 32 世界像素时沿有效覆盖中线取接缝，
      记录机位对与实测重叠宽度，标识符 `composition_narrow_overlap`
    - _Requirements: 10.1, 10.9_

  - [x] 13.3 移除默认路径的 `crop_to_valid_rectangle()` 裁切
    - 提前于阶段 3–5 执行，原因见执行记录
    - 默认路径改用 `mosaic.rs::mask_covered_bounds()` 的联合轴对齐外接边界，
      **删除** `stitching.rs:1399` 处 `valid_coverage < 0.995` 时对
      `crop_to_valid_rectangle()` 的调用；`crop_to_valid_rectangle()`
      函数本体（`stitching.rs:5785`）与旧对照路径（`stitching.rs:1989`）保留不改
    - 未覆盖像素保持完全透明，不做镜像/延拓/生成纹理；
      不把任何 Coverage_Mask 未覆盖的瓦片像素写入画布
    - 同步更新 `tests/focus-stack-quality-contract.mjs`：新增
      「默认路径不调用 `crop_to_valid_rectangle`」断言
    - _Requirements: 10.2, 10.3_

  - [x] 13.4 把默认锐化量归零，消除二次处理
    - 提前于阶段 3–5 执行，原因见执行记录
    - `stitching.rs:1379` 的 `RAW_EDITOR_FINAL_PANORAMA_SHARPEN_AMOUNT` 缺省值
      从 0.42 改为 **0.0**，使默认路径不再无条件执行 `sharpen_focus_tile_detail(.., 0.42)`
      （`stitching.rs:1386`）；`sharpen_focus_tile_detail()` 本体（`stitching.rs:8221`）
      与旧路径调用点（`stitching.rs:1976`、`stitching.rs:8160`）保留不改
    - 环境变量保留为诊断专用，实际值写入 `composition.final_sharpen_amount`
    - `RAW_EDITOR_FINAL_FOCUS_SHARPEN_AMOUNT` 保持现状（只影响旧对照路径）
    - 同步更新 `tests/focus-stack-quality-contract.mjs`：新增「最终锐化默认为 0」
      （`unwrap_or(0.0)`）断言
    - _Requirements: 11.6, 11.7_

  - [~] 13.5 实现整数平移快路径与输出 Ownership_Map
    - 复合变换与整数平移矩阵逐元素差 < 1e-9 时按整数偏移逐像素拷贝、不插值；
      否则用既有 `get_high_quality_interpolated_pixel()`（Catmull-Rom + 局部 min/max 钳制）
    - 输出 Ownership_Map 与最终画布同尺寸同坐标系，每个非透明像素的 owner 标识
      等于其所属 Virtual_Tile 的 Ownership_Map 对应位置标识
    - 记录 `composition.integer_translation_tiles`
    - _Requirements: 10.4_

  - [~] 13.6 画布上限、分块合成与输出编码
    - 新增 `MAX_OUTPUT_CANVAS_LONG_SIDE = 262_144`；超限拒绝写出，
      记录实测画布尺寸与上限，标识符 `canvas_long_side_exceeded`（Rejected）
    - 沿用 `StreamingMosaicStore` 的 1024px 分块，断言分块不改变输出像素尺寸、位深、
      Ownership_Map 与文件字节
    - 复用 `image_stack.rs::encode_srgb_image_stack()` 的 16 位 sRGB + ICC 写出；
      alpha 语义：未覆盖全透明、已覆盖全不透明，且写 alpha 不改变颜色通道值
    - 目标格式不支持 16 位或 alpha 时，在写出**之前**返回降级提示，
      记录实际位深与 alpha 保留状态，标识符 `output_bit_depth_downgraded` /
      `output_alpha_unsupported`
    - 复用 `canonicalize_image_stack_result()` / `write_preview_files()`：
      预览仅由最终结果规范化显示编码像素降采样得到，与最终结果携带同一结果标识
    - _Requirements: 10.2, 10.5, 10.6, 10.7, 10.10, 10.11, 14.5_
    - 执行记录：已加 `MAX_OUTPUT_CANVAS_LONG_SIDE = 262_144` 与 `compositor::reject_oversized_canvas`，默认路径在分配画布前拒绝超限画布并记录 `canvas_long_side_exceeded`（实测宽高、长边、上限），`is_canvas_rejection` 使回退渲染器不再重试。分块合成、alpha 写出、位深/alpha 降级提示与预览派生仍未做，保持 [~]。
    - 执行记录（alpha 与降级提示）：`StitchOutcome.coverage` 把分层路径的输出 Coverage_Mask 带到导出；`canonicalize_image_stack_result_with_coverage` 生成 RGBA16（未覆盖 alpha=0、已覆盖 alpha=65535，颜色通道与无 alpha 规范化逐值相同），TIFF（含元数据流式写出，新增 RGBA16/RGBA8 写出）与 PNG 写出 alpha，JPEG 丢弃 alpha 通道但颜色不变；预览仍由 RGB 派生，门禁 JPEG 输出不变。新增命令 `image_stack_output_notices` 在写出前返回位深 <16 与透明度无法保留的提示（`output_bit_depth_downgraded` / `output_alpha_unsupported`），前端尚未调用；导出时写回 Stack_Report 的实际位深与 alpha 状态、1024px 分块断言仍未做，保持 [~]。

  - [x] 13.7 删除 ownership 合成器环境变量并把旧合成器迁到设置项
    - 删除 `RAW_EDITOR_USE_OWNERSHIP_VIRTUAL_TILE_STITCHER`（其语义成为默认路径）
    - 把 `RAW_EDITOR_USE_STREAMING_VIRTUAL_TILE_MOSAIC`、`progressive_seam_stitcher`、
      `focus_stack_stitcher` 的选择迁到 `StackCompositorChoice` 设置项；
      四个函数本体行为不变
    - 同步更新 `tests/focus-stack-quality-contract.mjs`：新增「`Tile_Compositor`
      入口函数体内不出现 `std::env::var`」断言
    - _Requirements: 15.2, 15.3, 15.10_
    - 执行记录：删除 `RAW_EDITOR_USE_OWNERSHIP_VIRTUAL_TILE_STITCHER` 与 `RAW_EDITOR_USE_STREAMING_VIRTUAL_TILE_MOSAIC`，`StackCompositorChoice::resolve` 只读设置项；ownership 合成器的诊断开关收进 `OwnershipCompositorSwitches`，仅旧对照路径 `from_env` 读取，默认分层路径用全关的 `Default`，入口函数体内不再出现 `std::env::var`；契约新增对应断言。

  - [ ]* 13.8 属性测试：接缝只在双覆盖区且代价最低
    - **Property 53: 接缝只在双覆盖区且代价最低**
    - **Validates: Requirements 10.1**
    - 生成器 `arb_cost_grid(w, h, labels)`，≤8×8 重叠网格与穷举最小值比较

  - [x]* 13.9 属性测试：画布边界等于覆盖联合边界
    - **Property 54: 画布边界等于覆盖联合边界**
    - **Validates: Requirements 10.2**
    - 生成器 `arb_coverage_shape()`
    - 执行记录：P54 使用 100 个随机平移夹具，独立投影源四角得到联合边界，再逐像素核对 Coverage、采样原点和输出画布尺寸。

  - [x]* 13.10 属性测试：未覆盖像素保持透明且不被写入
    - **Property 55: 未覆盖像素保持透明且不被写入**
    - **Validates: Requirements 10.3**
    - 执行记录：P55 使用 100 个有间隙的平移，断言未覆盖像素 owner 为 `NO_OWNER`、颜色通道全零，且没有覆盖标记的像素落入间隙。

  - [x]* 13.11 属性测试：输出 ownership 与瓦片 ownership 一致
    - **Property 56: 输出 ownership 与瓦片 ownership 一致**
    - **Validates: Requirements 10.4**
    - 执行记录：P56 使用 100 个随机平移，通过 `assert_source_correspondence` 沿输出 owner 反查
      对应 Source_RAW 采样并逐像素核对 ownership 图例与采样坐标。

  - [x]* 13.12 属性测试：写入 alpha 不改变颜色通道
    - **Property 57: 写入 alpha 不改变颜色通道**
    - **Validates: Requirements 10.6**
    - 执行记录：`property_57_alpha_follows_coverage_without_changing_colour`（100 例，16 位 TIFF/PNG 写出再读回，alpha 与覆盖一致、颜色逐值等于无 alpha 规范化）与 `a_saved_tiff_with_alpha_validates_and_keeps_its_transparency`（生产保存路径）。

  - [x]* 13.13 属性测试：预览由最终结果派生且色差有界
    - **Property 58: 预览由最终结果派生且色差有界**
    - **Validates: Requirements 10.7**
    - 执行记录：预览改由纯函数 `derive_preview_images` 从规范化结果像素降采样得到，结果标识在写预览前生成并用于预览文件名（`{result_id}-detail.jpg` / `-interaction.jpg`），与存储结果同一标识。`property_58_previews_derive_from_the_result_with_bounded_roi_delta_e`（100 例）按生产 JPEG 质量往返后验证 2×/4× 预览的 64px ROI 低频均值 Delta_E00 ≤ 1.0。

  - [ ]* 13.14 属性测试：窄重叠沿中线取接缝
    - **Property 59: 窄重叠沿中线取接缝**
    - **Validates: Requirements 10.9**

  - [x]* 13.15 属性测试：画布超限拒绝输出
    - **Property 60: 画布超限拒绝输出**
    - **Validates: Requirements 10.11**
    - 执行记录：`property_60_oversized_canvas_is_rejected_with_its_measured_size`（100 例）与 `layered_compositor_rejects_an_oversized_canvas_before_loading_a_tile`（40000× 放大的 8px 瓦片，拒绝且不解码任何瓦片）。

  - [ ]* 13.16 属性测试：降内存措施不改变输出
    - **Property 89: 降内存措施不改变输出**
    - **Validates: Requirements 14.5**

  - [ ]* 13.17 单元测试：输出编码与降级提示
    - 扩展 `image_stack.rs` 现有 `encode_srgb_tiff` 测试，断言 16 位 sRGB + ICC（需求 10.5）
    - JPEG 的位深/alpha 降级提示在写出前返回（需求 10.10）
    - 默认构建下分层路径、组级色调、Quality_Gate 三者生效且入口不含 `env_var` 判断（需求 15.2）
    - 诊断开关默认关闭、旧单层路径标识符正确（需求 15.3, 15.10）
    - _Requirements: 10.5, 10.10, 15.2, 15.3, 15.10_
    - 执行记录：`output_fidelity_reports_bit_depth_and_alpha_downgrades_before_writing` 覆盖 JPEG 位深与 alpha 降级提示；16 位 sRGB + ICC 由既有 `exported_tiff_preserves_canonical_pixels_and_srgb_profile` 覆盖。默认构建入口无 env 判断与诊断开关默认关闭的单元测试未补，保持未勾选。

- [~] 14. 阶段 6 检查点
  - 验证画布边界逐值等于覆盖联合边界、未覆盖像素全透明、
    输出 ownership 与瓦片 ownership 一致；确保所有测试通过，遇到问题询问用户。

- [ ] 15. 阶段 7：Quality_Gate

  - [x] 15.1 实现 ROI 确定性选取与配对
    - `quality_gate.rs`：从输出 Ownership_Map 求 Owner_Region（4 连通，标号按区域最小
      世界坐标行优先序，与线程数无关）+ 区域内部距离变换
    - 候选 ROI 边长 512 原生像素，左上角按 256px 步长行优先枚举；
      接受条件为完全落在单个 Owner_Region 内、非透明比例 100%、四边距区域边界 ≥16px
    - 面积 ≥ 4×ROI 面积的区域至少取 1 个（取距离变换最大者，并列取行优先序最小者）；
      总数 > 256 时按 `stride = ceil(count / 256)` 等间隔抽取；
      < 32 时把步长降为 128px 再枚举一轮
    - 配对：用 owner 的 `member_to_anchor × tile_to_world × ResidualWarp` 复合逆变换
      映射 ROI 四角回源图坐标解码；**输出 ROI 不做任何重采样**，
      参考区域 ≤1 次重采样；相位相关残余对齐误差 > 0.5px 时该 ROI 全部测量项标为不可测量
      （`pairing_residual_alignment_exceeded`）
    - 所有 `HashMap` 遍历先 collect 再排序，保证 ROI 序列可复现
    - _Requirements: 11.1, 11.2, 11.17_
    - 执行记录：ROI 区域标号、距离变换、512/256px 枚举和源图四角复合逆映射已实现。

  - [x] 15.2 实现 Local_Scale 雅可比与有效像素总量基准
    - 对 ROI 内每 16px 采样点用解析复合矩阵中心差分求 `J`，
      `Local_Scale = sqrt(|det J|)`；判据为中位数 ≥0.98 且 `≥0.95` 的像素占比 ≥99%
    - 有效像素总量：分子为最终输出非透明像素计数；分母用与画布同尺寸位图对每个源图
      投影四边形扫描线填充置位后计数（重叠只计一次）；判据 ≥0.98×分母
    - `RAW_EDITOR_STACK_ACCEPTANCE_RENDER_SCALE < 1.0` 时把 `local_scale_median`
      标记为诊断模式并记录
    - _Requirements: 11.3, 11.4, 15.7_
    - 执行记录：16px 中心差分 Local_Scale 与唯一覆盖 scanline union 已实现。

  - [x] 15.3 实现 slanted-edge MTF50 度量与倾斜边 ROI 判定
    - 倾斜边判定：Canny + 概率霍夫（固定阈值）找 ≥128px 直线边、与最近像素轴夹角 ∈ [3°, 15°]、
      两侧低频亮度对比度 ≥ 满量程 20%、两侧各 32px 内无其他满足对比度条件的边
    - MTF50 实现（ISO 12233 风格，不引入新依赖）：逐行三次插值求边缘亚像素位置 →
      最小二乘直线拟合（残差 RMS > 0.5px 判不可测量
      `slanted_edge_line_fit_residual_exceeded`）→ 按法向距离重投影得超采样 ESF
      （过采样倍率 `round(1/tan θ)` 截断到 `[4, 16]`）→ 4 点滑动平均后一阶差分得 LSF →
      Hamming 窗 → 实数 DFT（零填充到 2 的幂）→ 归一化幅度谱 → 线性插值求 `f50`
    - `MTF50_Normalized = f50 / Local_Scale_median(ROI)`；
      参考区域用完全相同步骤（同过采样倍率规则与窗函数）
    - 判据：`≥ 0.93 × MTF50_Normalized_ref`
    - _Requirements: 11.5, 11.6_
    - 执行记录：Canny/Hough 倾斜边筛选、亚像素拟合、ESF/LSF/DFT MTF50 已实现。

  - [x] 15.4 实现归一化梯度能量与 Noise_Sigma 度量
    - 梯度能量：复用 `mosaic.rs::acutance()`（与 Glossary 的 Sharpness_Score 同一度量），
      在亮度通道对 ROI 全部采样点取平均后除以 `Local_Scale_median`；
      判据 `≥ 0.95 ×` 参考区域同一值
    - 平坦 ROI 判定：低频亮度标准差 ≤ 满量程 2%、不存在满足倾斜边条件的边、
      非透明比例 100%（否则 `roi_not_flat`）
    - Noise_Sigma：亮度通道减 σ = 2.0 原生像素高斯低通（分离卷积，核长 13）得高通，
      取 `sigma = 1.4826 × median(|hp − median(hp)|)`；参考区域在已重采样到 ROI 网格后
      用相同步骤测量；判据 `sigma_out / sigma_ref ∈ [0.85, 1.15]`
    - `composition.final_sharpen_amount > 0` 时把 `noise_sigma_ratio` 标记为诊断模式
      （不能用于验收通过）
    - _Requirements: 11.7, 11.8, 11.9_
    - 执行记录：acutance 梯度、平坦 ROI、13 tap sigma=2 高通 MAD Noise_Sigma 已实现。

  - [x] 15.5 实现单份 CIEDE2000 与 ROI 低频均值色差
    - sRGB EOTF 逆 → 线性 → XYZ(D65) → CIELAB(D65) → CIEDE2000（`kL = kC = kH = 1`）
    - 一份实现同时供需求 11.10（阈值 2.0）、9.8（1.5）、10.7（1.0）三处使用，
      把 11.6 中已有的 Delta_E00 调用点接到该实现
    - 判据：ROI 输出低频均值与配对参考区域低频均值 `ΔE00 ≤ 2.0`
    - _Requirements: 11.10_
    - 执行记录：复用 tone::delta_e00_rgb 的低频均值 Delta_E00 已实现。

  - [x] 15.6 实现边界笔画配准与 Sharpness_Confidence 覆盖率
    - 相邻 Owner_Region 公共边界按 Moore 邻域追踪（起点取边界上世界坐标行优先序最小像素），
      按弧长每 256 原生像素取 1 个测量点
    - 两侧各 16px 带内检测低频对比度 ≥ 满量程 15% 的边缘，
      仅两侧都检出且方向差 ≤10° 的作为同一笔画配对（否则 `boundary_no_pairable_edge`）；
      用与 slanted-edge 相同的逐行重心法求亚像素位置，取法向偏差
    - 判据：偏差 P95 ≤1.5px、最大 ≤3.0px
    - 输出级 Sharpness_Confidence 按输出 Ownership_Map 从各 Virtual_Tile 的
      `sharpness_confidence` 归属拷贝；判据：`< 0.05` 的像素占非透明像素比例 ≤1%
    - _Requirements: 11.11, 11.12_
    - 执行记录：Moore 边界、256px 采样、16px 笔画配准和置信度覆盖率已实现。
    - 2026-09-30：需求 11.12 的判据改为 owner 锐度覆盖率（`owner_sharpness_coverage`），由 15.23
      实现；本任务的置信度覆盖率降为诊断数值。

  - [x] 15.7 以「记录但不阻止」模式接入 Quality_Gate
    - 全部九项判据（`local_scale_median`、`local_scale_pixel_ratio`、
      `effective_pixel_count`、`mtf50_normalized`、`gradient_energy_normalized`、
      `noise_sigma_ratio`、`roi_delta_e00`、`boundary_stroke_alignment`、
      `owner_sharpness_coverage`（2026-09-30 前为 `sharpness_confidence_coverage`，见 15.23））
      在导出前执行，把实测值、阈值、可测量项数、
      不可测量项数、结论全部写入 `quality_gate`，但**不阻止导出**
    - 不可测量项既不计通过也不计失败，记录判据名称、世界坐标与原因标识符
    - 此任务**只加观测**，不改变导出行为
    - 执行记录：真实门禁 `langyuan-10` 输出 8788×10949、`wenyuan-10` 输出 8816×8666；两次报告均为 `tone.status=degraded`，边界最大 ΔE00 分别为 30.635、37.550，`quality_gate.verdict=not_run`，Quality_Gate 接线尚未完成。
    - 执行记录（合并色调修复 4398b8fa 后）：上一条的 30.635 / 37.550 测于不含色调修复的
      a2eec81d 基线。合并后渲染路径只多出 `residual_warp.rs` 的修正（两组门禁均为
      `residual_warp.identity=true`、0 个区域，不生效）与尚未接线的 `quality_gate.rs`，
      输出与 4398b8fa 一致；4398b8fa 实测两组仍为 `tone.status=degraded`，边界最大 ΔE00
      16.849 / 19.188，超限窗口 46 / 76（见检查点 12 记录），`quality_gate.verdict=not_run`。
    - _Requirements: 11.13, 11.16_
    - 执行记录（记录模式完成）：输出 ownership 已按「输出像素 → Capture_Station →
      Virtual_Tile 坐标 → 站 Ownership_Map → Source_RAW」反向解析，Coverage、Ownership
      与 Sharpness_Confidence 写入输出平面；单次门禁最多保留 1 个完整尺寸 Virtual_Tile。
      Langyuan 报告 `quality_gate.verdict=fail`，可测计数依次为
      `256/256/1/0/256/95/256/6/1`；Wenyuan 为 `233/233/1/0/233/70/233/0/1`。
      两组均记录九项阈值、失败/不可测世界坐标和原因，仍不阻止导出；输出 SHA256 与基线
      完全一致（`74942cfe…` / `86884740…`）。
    - 执行记录（最终代码门禁）：报告分别为
      `/private/tmp/raw-editor-gate/review6/task-final-langyuan/reports/stack-report-67ef811a-a209-46d2-834c-f35a4463edea.json`
      与 `/private/tmp/raw-editor-gate/review6/task-final-wenyuan/reports/stack-report-1abc7a93-b624-4e70-be97-74a9f87c35f9.json`。
      Langyuan 可测计数 `256/256/1/0/256/99/256/6/1`、不可测总数 6501；Wenyuan
      `231/231/1/0/231/44/231/0/1`、不可测总数 13470。两组 verdict 均为 `fail` 但仍只记录；
      输出 SHA256 仍为 `74942cfea3a1ede2…` / `86884740f93c876e…`。
    - 执行记录（`bee5516c` 后测量修正）：报告为
      `/private/tmp/raw-editor-gate/review-next/langyuan-report/reports/stack-report-e7f0dece-b455-4f16-83f2-5e298076ba3f.json`
      与 `/private/tmp/raw-editor-gate/review-next/wenyuan-report/reports/stack-report-d32bd569-01ac-4966-8a1a-5667b90212f4.json`。
      Langyuan 可测计数 `256/256/1/0/256/99/256/6/1`、不可测总数 6501；Wenyuan
      `231/231/1/0/231/44/231/0/1`、不可测总数 13470。MTF50 不可测原因、边界低对比度/方向不符、
      Textured_Pixel 与平坦像素计数及低置信度比例均写入报告。撤回 residual warp 后的最终输出 SHA256
      为 `74942cfea3a1ede2…` / `86884740f93c876e…`；实验曾启用 Wenyuan 一个 P95=9.5265px 区域，
      但输出变为 `e3f88653e9f02f4e…` 且无 post-warp P95 证据，故未接入生产。
      最终 Wenyuan 报告为 `/private/tmp/raw-editor-gate/review-next/wenyuan-final/reports/stack-report-1d9a8609-6ad4-4d07-b2aa-352170574a89.json`。
    - 执行记录（`65dfbbf0` 最终代码门禁）：报告为
      `/private/tmp/raw-editor-gate/rev65-corrected/lang-final/reports/stack-report-9f5cb0f1-6ea6-4179-8022-a9f18c0afe4d.json`
      与 `/private/tmp/raw-editor-gate/rev65-corrected/wen-final/reports/stack-report-1ed13320-e790-4125-a114-20c707ca70fb.json`。
      Langyuan 九项可测计数为 `256/256/1/0/256/99/256/6/1`，不可测分类为
      `content=0/0/0/256/0/157/0/6088/0`、`technical=0`；Wenyuan 可测计数为
      `256/256/1/0/256/73/256/0/1`，不可测分类为
      `content=0/0/0/256/0/183/0/12901/0`、`technical=0`。两组 MTF50 均为
      `not_applicable`，整体结论因可测量失败项为 `fail`，仍只记录不阻止导出。
      输出 SHA256 保持 `74942cfea3a1ede2…` / `86884740f93c876e…`。

  - [~] 15.8 切换 Quality_Gate 为阻止模式并接入拒绝路径
    - 在阶段 7 的「记录但不阻止」模式已在阆苑女仙 84 张上取得全部实测数字之后执行本任务
    - 任一判据存在未通过的可测量项 → 阻止写出最终结果文件、保留诊断预览与中间产物、
      Source_RAW 字节不变，返回指明未通过判据名称与对应 ROI 世界坐标的错误，
      标识符 `quality_gate_criterion_failed`（Rejected）
    - 判据结论按需求 11.14 的 (a)–(d) 四条规则确定（2026-09-29 修订）：`insufficient_evidence`
      阻止导出，标识符 `quality_gate_insufficient_evidence`（Rejected）；条件判据证据不够时为
      `not_applicable`，不阻止导出；两项整幅统计不受 8 项下限约束
    - Quality_Gate 成为导出的唯一闸门：导出函数只能从通过结论进入
    - 同步更新 `tests/focus-stack-quality-contract.mjs`：新增 Quality_Gate 阈值常量断言
      （`0.98 / 0.95 / 0.98 / 0.93 / 0.95 / [0.85, 1.15] / 2.0 / 1.5 / 3.0 / 0.01 / 8 / 0.20`
      全部以命名常量出现），以及判据名称集合与设计文档一致的断言、
      「`Quality_Gate` 入口函数体内不出现 `std::env::var`」断言
    - 执行记录：上述两次真实门禁均完成渲染并通过测试进程，但 Tone_Harmonizer 仍为 Degraded 且超过 1.5 ΔE00；阻止导出路径与 Quality_Gate 判据尚未接入，保持未完成。
    - _Requirements: 11.14, 11.15, 12.6, 15.2, 15.9_

  - [x] 15.9 属性测试：ROI 选取满足全部几何条件
    - **Property 61: ROI 选取满足全部几何条件**
    - **Validates: Requirements 11.1**
    - 执行记录：P61（100 cases）覆盖 ROI 区域完整性、透明度和边界余量。

  - [x] 15.10 属性测试：配对不重采样输出 ROI
    - **Property 62: 配对不重采样输出 ROI**
    - **Validates: Requirements 11.2**
    - 执行记录：P62（100 cases）验证复合逆映射、ROI 不变与残余对齐误差拒绝。

  - [x] 15.11 属性测试：Local_Scale 由雅可比确定且满足下界
    - **Property 63: Local_Scale 由雅可比确定且满足下界**
    - **Validates: Requirements 11.3**
    - 执行记录：P63（100 cases）验证恒等雅可比 Local_Scale 为 1。

  - [x] 15.12 属性测试：有效像素数不低于唯一覆盖面积基准
    - **Property 64: 有效像素数不低于唯一覆盖面积基准**
    - **Validates: Requirements 11.4, 15.7**
    - 执行记录：P64（100 cases）验证扫描线唯一并集与有效像素覆盖率下界。

  - [x] 15.13 属性测试：倾斜边与平坦 ROI 判定的充要性
    - **Property 65: 倾斜边与平坦 ROI 判定的充要性**
    - **Validates: Requirements 11.5, 11.8**
    - 生成器 `arb_slanted_edge(angle, blur)`、`arb_flat_patch(sigma)`
    - 执行记录：P65（100 cases）验证倾斜边阈值、拟合残差与平坦 ROI 互斥。

  - [x] 15.14 属性测试：MTF50_Normalized 不低于参考的 0.93 倍（含度量自检）
    - **Property 66: MTF50_Normalized 不低于参考的 0.93 倍**
    - **Validates: Requirements 11.6**
    - 执行记录：P66（100 cases）用合成高斯倾斜边完成解析 MTF50 自检。
    - 先用已知高斯模糊半径的合成倾斜边验证度量本身与解析 MTF50 一致，再验证判据

  - [x] 15.15 属性测试：归一化梯度能量不低于参考的 0.95 倍
    - **Property 67: 归一化梯度能量不低于参考的 0.95 倍**
    - **Validates: Requirements 11.7**
    - 执行记录：P67（100 cases）验证恒定 ROI 为零及局部尺度归一化。

  - [x] 15.16 属性测试：Noise_Sigma 比值落在规定范围（含度量自检）
    - **Property 68: Noise_Sigma 比值落在规定范围**
    - **Validates: Requirements 11.9**
    - 执行记录：P68（100 cases）用合成噪声完成 MAD Noise_Sigma 自检。
    - 先用已知标准差的合成高斯噪声验证高通 MAD 估计，再验证判据

  - [x] 15.17 属性测试：ROI 低频均值色差不超过 2.0（含 CIEDE2000 自检）
    - **Property 69: ROI 低频均值色差不超过 2.0**
    - **Validates: Requirements 11.10**
    - 执行记录：P69（100 cases）完成 CIEDE2000 对称性与同色为零自检。
    - 生成器 `arb_srgb_pair()`，验证 `ΔE(a, b) = ΔE(b, a)` 与 `ΔE(a, a) = 0`

  - [x] 15.18 属性测试：边界笔画配准偏差有界
    - **Property 70: 边界笔画配准偏差有界**
    - **Validates: Requirements 11.11**
    - 执行记录：P70（100 cases）验证 Moore 边界笔画配对的 P95、最大偏差和方向误差。

  - [ ] 15.19 属性测试：owner 锐度差额超限像素占比有界
    - **Property 71: owner 锐度差额超限像素占比有界**
    - **Validates: Requirements 11.12**
    - 执行记录：P71（100 cases）验证低置信覆盖率与 1% 阈值。
    - 2026-09-30 取消勾选：原 P71 调用的 `quality_gate::sharpness_confidence_coverage` /
      `sharpness_confidence_passes` 以全部非透明像素为范围，运行时并不使用（运行时用
      `quality_gate_runner::confidence_coverage_stats`），测的是平行实现；且需求 11.12 已改判据。
      按设计 Property 71 新文本重写，必须调用 15.23 中运行时实际使用的统计函数；两个旧辅助函数
      无其它调用者时删除。

  - [ ] 15.20 属性测试：测量项计数恒等且证据不足可判定
    - **Property 72: 测量项计数恒等且证据不足可判定**
    - **Validates: Requirements 11.13, 11.14**
    - 执行记录：P72（100 cases）验证可测量与不可测量计数恒等及证据不足判定。
    - 执行记录（2026-09-29）：P72（100 cases）按需求 11.13/11.14 的四条规则重写，覆盖
      `content_not_applicable` / `technical` 分类、`not_applicable` 序列化、20% 技术性不可测边界、
      条件/非条件判据的 8 项规则、整幅统计和整体结论优先级；失败种子无新增。
    - 执行记录（2026-09-30）：需求 11.14(c) 澄清为「有未通过的可测量项时按 11.15 判为未通过」。b0da0822 的实现把 Langyuan 边界笔画配准（6 项可测、5 项偏差 7–10 px）判成了 `not_applicable`，P72 与判定代码需按澄清后的规则修改后再勾选。

  - [x] 15.21 属性测试：阻止导出时不残留结果且保留诊断
    - **Property 73: 阻止导出时不残留结果且保留诊断**
    - **Validates: Requirements 11.15, 12.6, 12.10**
    - 执行记录：P73（100 cases）验证失败报告保留判据、ROI 与不可测量原因。

  - [x] 15.22 属性测试：Quality_Gate 的 ROI 集合与结论可复现
    - **Property 74: Quality_Gate 的 ROI 集合与结论可复现**
    - **Validates: Requirements 11.17**
    - 执行记录：P74（100 cases）验证同 Ownership_Map 输入的 ROI 集合与标签距离完全一致。

  - [ ] 15.23 把需求 11.12 判据改为 owner 锐度覆盖率（2026-09-30 用户确认）
    - 按设计 Focus_Fuser 第 11 条在单元级计算 Owner_Sharpness_Shortfall：`SharpnessCellEvidence`
      已有 `winner_score` / `owner_score` / `candidate_count`，补 `disagreement`（该单元
      `StationFusion::disagreement`）；不得改变任何 owner 决策
    - 站级逐像素值与 `confidence`、`textured` 使用同一个 `focus_cell_plane_to_pixels` 映射；
      输出级值在 `panorama_stitching.rs` 现有的输出 ownership 反查循环中一并归属拷贝，
      不新增整幅 `f32` 平面（`u8` 标志平面或流式计数均可）
    - 判据标识符 `sharpness_confidence_coverage` → `owner_sharpness_coverage`，同步
      `quality_gate.rs`、`quality_gate_runner.rs`、`test_support.rs`、`properties.rs`、`report.rs`
      与 `tests/focus-stack-quality-contract.mjs`；阈值用命名常量（shortfall 上限 0.05、占比上限
      0.01），数值不变
    - `unresolved_pixel_count`：非透明但反查不到 owner 单元证据的像素单独计数，不再落入平坦像素；
      计数大于 0 时判据为技术性不可测、结论为证据不足。`65dfbbf0` 门禁报告中这类像素 Langyuan
      19 个、Wenyuan 95 个（有效像素分项分子之和与非透明像素数之差），先查明原因并修复反查，
      使两组为 0。另有一处漏计：站 Ownership_Map 为 `NO_OWNER` 时，反查里的
      `usize::from(raw_local).saturating_sub(1)` 取到下标 0，把像素记给了该站第 1 帧；
      这也是反查失败，必须计入 unresolved，不得归属任何帧
    - Sharpness_Confidence 的两个子集低置信占比与 `confidence_scores` 分布保留为诊断字段；
      新增 shortfall > 0.05 像素的 20 桶直方图、其中 disagreement > 0.2 的像素数、按
      Capture_Station 分项
    - 只改报告，两组门禁输出 SHA256 必须保持 `74942cfe…` / `86884740…`
    - _Requirements: 11.12, 11.13, 11.14_

- [~] 16. 阶段 7 检查点
  - 验证三个度量的已知答案自检通过、同输入重复运行的 ROI 序列与结论完全相同；
    确保所有测试通过，遇到问题询问用户。

- [ ] 17. 阶段 8：资源边界、诊断与验收

  - [x] 17.1 实现内存门槛解析与峰值 RSS 采样
    - 新增 `MEMORY_THRESHOLD_DEFAULT_BYTES = 24 GiB`、`MEMORY_THRESHOLD_MIN_BYTES = 4 GiB`、
      `MEMORY_THRESHOLD_PHYSICAL_RATIO = 0.75`；物理内存取 `sysinfo::System::total_memory()`
    - 解析为纯函数：用户配置 → `clamp(值, 4 GiB, 0.75×物理)` / `UserConfigured`；
      自动校准 → 上界 / `AutoCalibrated`；否则 `clamp(24 GiB, 下界, 上界)` / `Default`；
      上界 < 下界时取上界并标 `AutoCalibrated`
    - 后台线程以 500 ms（≥1 Hz）读取本进程 RSS，维护最大值与采样计数，
      写入 `resources.{peak_rss_bytes, rss_sample_count, memory_threshold_bytes,
memory_threshold_source, physical_memory_bytes}`
    - 保持 `BlendMode::FocusStack` 的 `render_scale = 1.0`
      （`memory_safe_panorama_render_scale()` 只作用于 `BlendMode::Panorama`），
      这是需求 11.3 的必要前提
    - _Requirements: 14.1, 14.2_
    - 执行记录：`resources.rs` 实现确定性门槛解析、500 ms RSS 采样、注入 RSS 源和共享取消标志；运行开始启动，报告边界写入峰值 RSS、样本数、门槛、物理内存与来源。两组门禁来源均为 `default`，峰值约 9.23 / 9.34 GB。

  - [x] 17.2 实现内存超限中止与取消路径
    - 超门槛立即设置取消标志 → 各阶段在下一个检查点退出 → 删除本次运行临时文件 →
      写报告（`memory_threshold_exceeded`、门槛、实测峰值、采样次数），不写部分结果文件
    - 取消：5 秒内停止解码新源图、删除临时文件、不写部分结果、
      源图字节与已有 Virtual_Tile 缓存条目保持不变，标识符 `run_cancelled_by_user`
    - 进度以 ≤2 秒间隔更新阶段名称、已完成机位数、总机位数；取消请求 1 秒内确认
    - 断言无出站网络请求，`resources.network_requests` 计数写入报告
    - _Requirements: 14.3, 14.7, 14.8, 14.9_
    - 执行记录：FocusRssGuard 在解码、合成和质量门边界检查取消标志；超限时在最终文件写入前
      返回，报告写入 `memory_threshold_exceeded`、门槛、峰值和样本数；注入 RSS 单测覆盖取消。
    - 执行记录（新一轮 generation 取消）：新请求发布 generation 后，旧 run 在解码、匹配、合成和
      Quality_Gate 检查点返回 `run_cancelled_by_user`，排队的新 run 在获得全局锁前检测过期并在
      小于 1 秒内确认；`newer_generation_is_detected_without_cancelling_direct_runs` 与
      `superseded_queued_generation_is_acknowledged_within_one_second` 覆盖该路径。

  - [~] 17.3 把 `mosaic_diagnostics.rs` 从 `#[cfg(test)]` 提升为设置项驱动
    - 从 `#[cfg(test)]` 改为正常编译，由设置项 `stack_diagnostics.output_dir` 控制
      （替代 `RAW_EDITOR_MOSAIC_DIAGNOSTICS` 环境变量；测试可继续用环境变量覆盖）
    - 输出扩展到 7 类：机位成员集合、每帧变换、局部残差场、Ownership_Map、
      Sharpness_Confidence、Coverage_Mask、低频色调场，每项携带 Capture_Station 标识
    - 把现有 `capture_crop()` 记录的「最大全有效矩形」替换为实际使用的联合边界，
      并记录最终有效裁切区域左上角、宽、高
    - 诊断缓冲用 `Option<Box<...>>`，关闭时保持 `None`（零分配、零写入）
    - `RAW_EDITOR_STACK_GROUP_DIAGNOSTICS` 保持现状（仅影响可读打印）
    - _Requirements: 13.1, 13.2, 13.5_
    - 执行记录：`diagnostics::DiagnosticsRecorder` 已实现：关闭时为 `None`（一个空指针），不运行任何条目生产闭包、不写文件；开启时七类条目（成员、每帧变换、局部残差场、Ownership_Map、Sharpness_Confidence、Coverage_Mask、低频色调场）按站位命名即时写出并列入 `diagnostics.json`，记录最终有效裁切的左上角与宽高；首次写入失败后停止后续写入并以 `diagnostics_write_failed` 指明目录。`stack_diagnostics.output_dir` 设置项与管线接线、替换 `capture_crop` 仍未做，保持 [~]。

  - [~] 17.4 实现诊断 ROI 导出与像素级溯源查询
    - 长边 ≤4096 世界像素的 ROI：60 秒内导出该 ROI 内每个候选 Source_RAW 的重采样结果、
      选择掩膜、逐帧 Sharpness_Score 与最终 ownership，各项同尺寸同原点
    - 像素溯源：给定输出非透明像素坐标，查 `ownership` + `legend` 返回 Capture_Station 标识、
      owner 绝对路径、Sharpness_Confidence、Coverage_Mask 取值
    - 只写用户指定目录；目录不存在/不可写/写入失败 → 停止后续诊断写入、
      保留最终输出与报告、提示指明失败与目标目录，标识符 `diagnostics_write_failed`
    - ROI 长边 >4096 / 在裁切区外 / 与覆盖无交集 → 拒绝该次导出、不写部分结果、
      提示指明原因，标识符 `diagnostics_roi_invalid`
    - _Requirements: 13.3, 13.6, 13.7, 13.8_
    - 执行记录：新增 `stack_pipeline/diagnostics.rs`：`validate_diagnostic_roi`（长边 >4096、完全在输出外、与覆盖无交集时以 `diagnostics_roi_invalid` 与原因拒绝，写入前判定）、`pixel_provenance`（站位、owner 绝对路径、Sharpness_Confidence、Coverage_Mask）与 `export_roi_planes`（只写用户目录，先写隐藏暂存目录再一次重命名发布，失败时删除暂存并以 `diagnostics_write_failed` 指明目录）。候选 Source_RAW 重采样、选择掩膜、逐帧 Sharpness_Score、60 秒时限与管线接线尚未完成，保持 [~]。

  - [x] 17.5 实现 `stack_acceptance_harness`
    - 复用 `panorama_reference_acceptance.rs` 的 `#[ignore]` + 环境变量素材路径 +
      生产 RAW 加载器 + JSON 清单骨架；**不接受任何参考图像或外部单应性清单参数**
    - 入口变量 `RAW_EDITOR_STACK_ACCEPTANCE_SOURCE_DIR` /
      `RAW_EDITOR_STACK_ACCEPTANCE_REPORT_DIR`；枚举
      `DSC_3680.NEF`–`DSC_3763.NEF` 共 84 张走完整默认 Stack_Pipeline，
      把 Stack_Report JSON 写入指定临时目录
    - 断言：84 张全部入组且 `isolated.len() == 0`、机位图单一连通、
      `quality_gate.verdict == "pass"` 且未通过判据数 0、
      `opaque_pixels / union_projected_pixels ≥ 0.98`、
      `peak_rss_bytes ≤ memory_threshold_bytes`、`network_requests == 0`、
      `fusion[*].solver_status == "graph_cut"`、素材目录文件列表与各文件 SHA-256
      运行前后一致
    - 失败时保留已产出报告并记录未满足判据的标识符、实测值与阈值
    - 新增 `npm run stack-acceptance:check` 脚本入口
    - _Requirements: 15.4, 15.5, 15.6, 15.7, 15.8, 15.12_
    - 执行记录：已添加 `#[ignore]` 的 `stack_acceptance_harness`、纯 verdict 函数与
      `npm run stack-acceptance:check`；按要求未在 84 张 NEF 上运行。

  - [ ]* 17.6 属性测试：诊断输出完整且与输出同坐标系
    - **Property 81: 诊断输出完整且与输出同坐标系**
    - **Validates: Requirements 13.1, 13.2**

  - [ ]* 17.7 属性测试：ROI 诊断导出同尺寸同原点
    - **Property 82: ROI 诊断导出同尺寸同原点**
    - **Validates: Requirements 13.3**

  - [x]* 17.8 属性测试：诊断关闭时零分配零写入
    - **Property 83: 诊断关闭时零分配零写入**
    - **Validates: Requirements 13.5**
    - 执行记录：`property_83_disabled_diagnostics_allocate_and_write_nothing`（100 例）与 `a_failed_diagnostic_write_stops_later_writes_and_names_the_directory`。

  - [x]* 17.9 属性测试：像素级溯源查询一致
    - **Property 84: 像素级溯源查询一致**
    - **Validates: Requirements 13.6**
    - 执行记录：`property_84_pixel_provenance_matches_the_output_planes`（100 例）。

  - [x]* 17.10 属性测试：无效诊断 ROI 拒绝且不写部分结果
    - **Property 85: 无效诊断 ROI 拒绝且不写部分结果**
    - **Validates: Requirements 13.8**
    - 执行记录：`property_85_invalid_diagnostic_rois_are_refused_without_writing`（100 例，独立判定三种拒绝条件，并验证拒绝时目录为空、接受时只发布一个 ROI 目录）。

  - [x]* 17.11 属性测试：内存门槛解析确定
    - **Property 86: 内存门槛解析确定**
    - **Validates: Requirements 14.2**
    - 执行记录：P86 使用 100 cases，逐例核对低物理内存上界优先、配置/自动校准/默认来源和截断值，
      并重复解析比较 `MemoryThreshold`。

  - [ ]* 17.12 属性测试：内存超限中止且不残留
    - **Property 87: 内存超限中止且不残留**
    - **Validates: Requirements 14.3**

  - [ ]* 17.13 属性测试：取消后不残留且状态可追溯
    - **Property 91: 取消后不残留且状态可追溯**
    - **Validates: Requirements 14.8**

  - [ ]* 17.14 属性测试：Stack_Report schema 完整且与返回值一致
    - **Property 93: Stack_Report schema 完整且与返回值一致**
    - **Validates: Requirements 1.7, 2.9, 5.9, 6.10, 11.16**
    - 生成器 `arb_stack_report()`

  - [x]* 17.15 属性测试：Acceptance_Harness 判定与阈值一致
    - **Property 94: Acceptance_Harness 判定与阈值一致**
    - **Validates: Requirements 15.12**
    - 执行记录：P94 使用 100 cases，纯函数按 84 张入组/连通、Quality_Gate 失败数和 0.98 联合面积
      三项条件判定，并与独立期望逐例一致。

  - [ ]* 17.16 集成测试：时限类判据
    - Stack_Report 在成功/降级/拒绝三条路径上 30 秒内写出（需求 10.8）
    - 进度事件间隔 ≤2 秒、取消确认 ≤1 秒、取消完成 ≤5 秒（需求 14.7, 14.8）
    - 诊断 ROI 导出 ≤60 秒（需求 13.3）
    - 诊断目录不可写注入（需求 13.7）
    - _Requirements: 10.8, 13.3, 13.7, 14.7, 14.8_

- [~] 18. 阶段 8 检查点
  - 在阆苑女仙 84 张上跑 `stack_acceptance_harness`，验证峰值 RSS ≤ 生效门槛
    且全部通过条件满足；确保常规回归套件全部通过，遇到问题询问用户。

- [~] 19. 回归收口：常规回归契约补全
  - 补齐 `tests/focus-stack-quality-contract.mjs` 中前面各阶段尚未落地的常量与源码级断言，
    使 design.md「常规回归契约的扩展」11 条全部生效，并复核
    `tests/image-stack-preview-contract.mjs` 的三向一致断言仍锚定 500
  - 复核 `docs/focus-stack-reference-validation.md` 四项失败在
    「失败项 → 自动判定断言」映射表中的每一行都有对应的已实现断言（属性测试或 harness 断言），
    缺失项补写测试代码
  - 全部断言在不读取任何 RAW、不访问网络的条件下执行
  - _Requirements: 15.9_

## Notes

- 标注 `*` 的子任务为可选（属性测试、单元测试、集成测试），可为 MVP 跳过；
  顶层任务与检查点不带 `*`。
- 每条 Correctness Property 恰好对应一个属性测试任务，不拆分也不合并；
  Rust 侧用 `proptest`、`ProptestConfig { cases: 100, .. }`，失败用例写入
  `proptest-regressions/` 并纳入版本控制；测试首行注释必须是
  `// Feature: layered-camera-group-focus-stitching, Property N: ...`。
- 测试分层：任务 1–17 中的 `*` 子任务均为**不读 RAW、不联网**的常规回归；
  只有任务 17.5 的 `stack_acceptance_harness` 需要阆苑女仙 84 张 NEF，
  以 `#[ignore]` 标记手动触发。
- 「先加观测，再改行为」体现在两处成对任务：1.2/1.3（报告与 ledger 只记录，不改回退行为）
  与 15.7/15.8（Quality_Gate 先记录不阻止，取得实测数字后再切换为阻止模式）。
- `docs/focus-stack-reference-validation.md` 四项失败的四个代码级根因分别由
  13.3（`crop_to_valid_rectangle` 裁掉画框）、13.4（无条件
  `sharpen_focus_tile_detail(.., 0.42)` 的二次处理）、9.4（`FOCUS_GROUP_CONSENSUS_MAX_ERROR_RATIO`
  与 9.8 的 `FOCUS_GROUP_MAX_CENTER_CORRECTION_RATIO` 门槛过松）、
  13.2（接缝代价缺少到有效覆盖边界的距离项）四个可独立验证的任务承担；
  需求 7.2 被违反的预剔除由 9.6 单独承担。
- 涉及改常量或改函数签名的任务都包含「同步更新相关断言」子项：
  `tests/focus-stack-quality-contract.mjs` 的源码级常量与函数存在性断言、
  `mosaic.rs` 的 24 个单元测试、`seam_cut::cut_grid` 与穷举最小值对齐的测试
  必须在每个检查点保持通过。
- 旧路径函数本体（`focus_stack_stitcher`、`progressive_seam_stitcher`、
  `detail_preserving_mosaic`、`crop_to_valid_rectangle`、`sharpen_focus_tile_detail`）
  全程不改行为，只切走默认路径调用点。
- design.md 阶段 9 的文档更新（`docs/focus-stack-reference-validation.md`、
  `docs/focus-stack-algorithm.md`、README 图像堆栈段落）属于非编码任务，未列入本计划；
  其中唯一的编码部分（回归契约补全）已合并为任务 19。README 与前端契约中
  2–200 → 2–500 的数字一致性因受测试守卫，保留在任务 1.1。

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["1.4", "1.5", "3.1", "3.2", "3.3"] },
    { "id": 2, "tasks": ["3.4", "3.7", "3.5", "3.6"] },
    { "id": 3, "tasks": ["5.1", "5.2"] },
    { "id": 4, "tasks": ["5.3", "5.4", "5.5"] },
    { "id": 5, "tasks": ["5.6", "5.7", "5.8", "5.9", "5.10", "5.11", "5.12", "5.13", "5.14", "5.15"] },
    { "id": 6, "tasks": ["7.1", "7.4", "7.6"] },
    { "id": 7, "tasks": ["7.2", "7.5", "7.7"] },
    { "id": 8, "tasks": ["7.3", "7.8", "7.10"] },
    { "id": 9, "tasks": ["7.9", "7.11", "7.12"] },
    {
      "id": 10,
      "tasks": [
        "7.13",
        "7.14",
        "7.15",
        "7.16",
        "7.17",
        "7.18",
        "7.19",
        "7.20",
        "7.21",
        "7.22",
        "7.23",
        "7.24",
        "7.25",
        "7.26",
        "7.27",
        "7.28",
        "7.29",
        "7.30",
        "7.31"
      ]
    },
    { "id": 11, "tasks": ["9.1", "9.3", "9.6"] },
    { "id": 12, "tasks": ["9.2", "9.4", "9.7"] },
    { "id": 13, "tasks": ["9.5", "9.8", "9.10"] },
    { "id": 14, "tasks": ["9.9", "9.11"] },
    {
      "id": 15,
      "tasks": [
        "9.12",
        "9.13",
        "9.14",
        "9.15",
        "9.16",
        "9.17",
        "9.18",
        "9.19",
        "9.20",
        "9.21",
        "9.22",
        "9.23",
        "9.24",
        "9.25",
        "9.26",
        "9.27",
        "9.28",
        "9.29",
        "9.30",
        "9.31"
      ]
    },
    { "id": 16, "tasks": ["11.1", "11.4", "11.7"] },
    { "id": 17, "tasks": ["11.2", "11.5"] },
    { "id": 18, "tasks": ["11.3", "11.6"] },
    {
      "id": 19,
      "tasks": [
        "11.8",
        "11.9",
        "11.10",
        "11.11",
        "11.12",
        "11.13",
        "11.14",
        "11.15",
        "11.16",
        "11.17",
        "11.18",
        "11.19"
      ]
    },
    { "id": 20, "tasks": ["13.1", "13.2", "13.6"] },
    { "id": 21, "tasks": ["13.3", "13.5"] },
    { "id": 22, "tasks": ["13.4", "13.7"] },
    {
      "id": 23,
      "tasks": ["13.8", "13.9", "13.10", "13.11", "13.12", "13.13", "13.14", "13.15", "13.16", "13.17"]
    },
    { "id": 24, "tasks": ["15.1", "15.3", "15.5"] },
    { "id": 25, "tasks": ["15.2", "15.4", "15.6"] },
    { "id": 26, "tasks": ["15.7"] },
    { "id": 27, "tasks": ["15.8"] },
    {
      "id": 28,
      "tasks": [
        "15.9",
        "15.10",
        "15.11",
        "15.12",
        "15.13",
        "15.14",
        "15.15",
        "15.16",
        "15.17",
        "15.18",
        "15.19",
        "15.20",
        "15.21",
        "15.22"
      ]
    },
    { "id": 29, "tasks": ["17.1", "17.3"] },
    { "id": 30, "tasks": ["17.2", "17.4"] },
    { "id": 31, "tasks": ["17.5"] },
    {
      "id": 32,
      "tasks": ["17.6", "17.7", "17.8", "17.9", "17.10", "17.11", "17.12", "17.13", "17.14", "17.15", "17.16"]
    }
  ]
}
```
