# Focus Stack 当前架构与拆分边界

本文只描述当前工作区可见的生产入口、模块职责和后续拆分边界。模块拆分以
`.kiro/specs/layered-camera-group-focus-stitching/tasks.md` 的 N0–N9 为准；未接入生产的研究模块不算作
当前架构的一部分。

本文按 `v1` 分支 2026-10-29 的 N1-10 代码基线修订（长文件拆分合并之后）。全部 facade、子模块、行数与职责见
[`source-layout.md`](source-layout.md)，该表由 `scripts/source-layout.py` 从源码生成；新增模块的行数以当前工作区为准；
历史记录中的行号和临时报告路径只用于追溯，不能作为当前模块定位或验收依据。

## 当前入口

`src-tauri/src/panorama_stitching.rs` 是 Rust 堆栈入口，负责输入预检、源图准备、特征匹配、全局关系、
机位求解、focus-stack 执行流程、取消、缓存和最终输出发布。N1-10 之后它是 936 行的 facade：共享常量、`#[path]`
兄弟模块声明、兼容重导出和两个 Tauri 命令（`stitch_panorama`、`save_panorama`）；实现位于
`src-tauri/src/panorama_stitching/` 的 38 个生产子模块，总编排函数 `stitch_images_with_options`（约 3,820 行，
单函数）在 `stitch_images.rs`。N1 完成前仍不把其中的实验入口视为稳定 API。

候选 pair 枚举已移到 `src-tauri/src/panorama_feature_pairs.rs`。该模块只负责输入顺序/自然文件名排序、
邻域候选、受限视觉检索和大堆栈候选扩展；完整特征提取、几何验证和后续 solver 仍由入口编排。入口通过
私有 facade import 保持原调用名，未改变候选阈值、排序规则或回退路径。

N1-5 又新增 `panorama_focus_feature_prep.rs` 和 `panorama_match_contract.rs`。前者负责 bounded analysis
image 的 focus 特征、foreground range/mask、top feature 和 foreground predicate；后者负责 canonical relation
方向、空间支持、homography scale、station local scale 以及 adjacent same-focal pair 合同。两者分别为 279
和 262 行。

N1-6 将 `match_image_pair_with_seed` 的匹配执行体拆成四个阶段模块：
`panorama_match_candidates.rs`（候选生成与 seeded/coarse 候选，97 行）、
`panorama_match_geometry.rs`（投影、背景过滤、RANSAC、模型选择和 inlier 验证，584 行）、
`panorama_match_focus_rescue.rs`（mixed-focal、focus overlap 和 adjacent focus-layer rescue，270 行）以及
`panorama_match_result.rs`（统一 `MatchInfo` 结果组装，28 行）。入口只保留候选阶段调用、几何阶段调用、
focus rescue 分支和最终结果编排；阈值、坐标、回退顺序、日志语义与 `match_image_pair` 调用合同保持不变。

N1-7 把 station 关系门、闭环、拓扑、pose 和 solver 从入口拆成九个私有子模块。它们都是
`panorama_stitching` 的 `#[path]` 子模块，入口用 facade import 保持原调用名；依赖方向只有子模块读取入口的
常量/共享 helper 和前序 station 模块，入口不承载这些实现：

| 模块                                 | 行数 | 职责                                                                                                                                    |
| ------------------------------------ | ---: | --------------------------------------------------------------------------------------------------------------------------------------- |
| `panorama_station_relation_gate.rs`  |  566 | Station_Relation 证据类型、Virtual_Tile coverage 视图、requirement-6 测量、拒绝原因和共享 gate、overlap/spatial support 与凸包/裁剪几何 |
| `panorama_station_closure.rs`        |  737 | Closure_Optimizer 的 translation/affine/projective robust solve、M-estimator 权重和收敛判据                                             |
| `panorama_station_closure_gate.rs`   |  637 | 闭环采样点、corner/overlap 修正上限、closure acceptance、direct-pair P95 和保持连通的 relation 拒绝                                     |
| `panorama_station_closure_refine.rs` |  749 | 从最大 inlier 生成树种子出发的闭环 driver 和 ClosureReport 记录                                                                         |
| `panorama_station_topology.rs`       |  258 | 生成树 key、pose-free topology 观测、连通分量和断连错误                                                                                 |
| `panorama_station_pose_seed.rs`      |  339 | translation pose 种子、长边 roll 校正和 Local_Scale pose candidate guard                                                                |
| `panorama_station_solver.rs`         |  899 | Source_RAW 与 Virtual_Tile 共用的 station pose solver 主流程                                                                            |
| `panorama_station_solve_entry.rs`    |  106 | 按证据类型进入共享 solver 的入口                                                                                                        |
| `panorama_station_residual.rs`       |  177 | 从已接受 relation 对应点构建 Residual_Warp 证据                                                                                         |

这一步只移动代码：阈值、坐标、回退顺序、日志和报告字段均未改变，`panorama_stitching.rs` 从 36,315 行降到
32,184 行。

N1-8 继续拆出 focus grouping：`panorama_station_grouping.rs`（721 行）负责 pairwise station evidence、局部
bracket 运动判定、verified scan overlap、按运动拆分连通组件和 `StationGroupingOutcome`；
`panorama_capture_groups.rs`（464 行）负责 `FocusCaptureGroup`、binding evidence、bracket pose 锁定、capture
group 构建、anchor 规范化和按 capture group 的渲染顺序。移出的 1,089 行非空代码与新模块逐行一致，facade 降到
31,077 行。警告集合与 `v1` 一致，release 全量库测试 `901 passed, 0 failed, 40 ignored`，grouping 相关定向测试
通过，兰亭与马远 PSD 10% 回归仍与 `v1` 基线逐字节一致。`absolute_source_path`、输入预检 helper 和
Virtual_Tile 关系匹配仍在入口，留给后续拆分。

N1-9 把 `src-tauri/src/panorama_utils/stitching.rs`（`v1` 上 18,898 行）拆成 574 行的 facade 和
`panorama_utils/stitching/` 下的 34 个子模块。facade 只保留共享常量、`use`、模块声明、兼容重导出、两个
oracle 模块声明与 `reset_final_world_owner_oracle`、四个跨区共享的像素 helper（`luminance`、
`focus_stack_pixel_is_canvas_like`、`focus_stack_pixel_is_tone_foreground`、`median_f32`）以及两个小型内联契约
测试模块。移出的条目按原可见性在 facade 重导出，外部 `stitching::X` 路径不变；私有条目改为 `pub(super)`，
原 `pub(super)` 条目改为 `pub(in crate::panorama_utils)`；函数体只把 43 处 `super::<兄弟模块>::` 路径改写为
`crate::panorama_utils::<兄弟模块>::`。生产子模块如下：

| 模块                         |  行数 | 职责                                                                                                     |
| ---------------------------- | ----: | -------------------------------------------------------------------------------------------------------- |
| `canvas_maps.rs`             |   345 | `NO_OWNER`、ownership/coverage/confidence map、source provenance、残差坐标映射和带 focus warp 的画布尺寸 |
| `virtual_tiles.rs`           |   350 | Virtual_Tile 与 focus virtual-tile 几何、native station 几何                                             |
| `exposure.rs`                |   431 | 曝光增益与补偿、重叠区曝光估计                                                                           |
| `station_seam_sampling.rs`   |   539 | station seam plan 类型、relation/owner 判定、seam 采样、局部 tone 与边缘误差                             |
| `station_seam_plan.rs`       |   793 | station seam plan 构建、cross/series 查询与 transition 候选                                              |
| `station_seam_apply.rs`      |   639 | station seam plan 应用、owner 拷贝、owner 岛清理和 owner trace                                           |
| `progressive_seam.rs`        |   680 | 渐进接缝 stitcher、平面扫描纸色中位数和 photometric model 应用                                           |
| `panorama_seam_band.rs`      |   337 | 全景接缝带 detail alpha 与混合                                                                           |
| `layered_compositor.rs`      |   441 | 分层 Virtual_Tile compositor 入口、tile payload/source、finishing 开关和 tile 边界距离                   |
| `tile_ownership_stitcher.rs` | 1,363 | `focus_tile_ownership_stitcher_with_finishing` 主体（单函数 1,302 行）                                   |
| `focus_layers.rs`            |   496 | focus layer 渲染、band inverse 与 source sampling                                                        |
| `focus_color.rs`             |   639 | focus 颜色校正估计与应用                                                                                 |
| `focus_decision.rs`          |   304 | focus 分析尺寸、score map、决策 mask 和画布切换抑制                                                      |
| `focus_masks.rs`             |   492 | 二值 mask 形态学、前景孔洞与连通域、背景参考 mask 和 focus 像素放置                                      |
| `focus_alignment.rs`         |   909 | focus layer 平移与 patch 配准、station 帧 native 配准和前景对齐                                          |
| `focus_foreground.rs`        |   533 | 前景抑制、overlap/edge mask、前景与 owner 标记                                                           |
| `focus_analysis.rs`          |   429 | focus 分析层、决策、cell 测量与 cell 到像素的展开                                                        |
| `filters.rs`                 |   511 | RGB 缩放与下采样、金字塔辅助、masked box blur 和 multiband blend                                         |
| `canvas_fill.rs`             |   441 | 有效矩形裁切、无效像素 run 填充和 focus 画布边距填充                                                     |
| `focus_tone.rs`              |   717 | focus 背景 tone 协调（含 owner 与参考版本）                                                              |
| `focus_seam_blend.rs`        |   489 | focus seam band 混合与 patch 外硬选择                                                                    |
| `focus_stack_entry.rs`       |   415 | focus-stack stitcher 入口、tile render 记录、平面采样和 station 成员排序                                 |
| `focus_stack_stitcher.rs`    |   972 | 全局 focus-stack compositor 主体与可选 tile 锐化                                                         |
| `seam_path.rs`               |   388 | 自适应与 pairwise DP 最小代价接缝                                                                        |
| `sampling.rs`                |   161 | homography warp 与双线性/双三次采样                                                                      |

测试专用子模块（均为 `#[cfg(test)]`）：`station_stroke_match.rs`（491）、`station_seam_diagnostics.rs`（540）、
`test_tone_capture.rs`（144）、`final_world_owner_runner.rs`（1,067，单函数 `run_final_world_owner_oracle`）、
`score_noise_probes.rs`（552）、`score_noise_diagnostics.rs`（883），以及原样外置的内联测试模块
`interpolation_tests.rs`（1,480）、`station_degradation_wiring_tests.rs`（316）和 `focus_fusion_test_access.rs`
（15）。超过 1000 行的 `tile_ownership_stitcher.rs`、`final_world_owner_runner.rs` 和 `interpolation_tests.rs`
分别留给按阶段拆函数和 N2 测试拆分（`interpolation_tests.rs` 已在 N2 按主题拆分）。

N1-10 用同样的方法拆分了其余超过 1000 行的文件，按 facade 列出：`panorama_stitching.rs`（31,128 → 936 行，
38 个生产子模块 + 5 个外置测试模块）、`panorama_utils/mosaic.rs`（7,079 → 277，13 + 2）、`scan_mosaic.rs`
（2,861 → 264，7）、`processing.rs`（1,299 → 98，2 + tests），以及 `stack_pipeline` 中的 compositor、degradation、
intra_station、residual_warp、joint_warp、quality_gate、virtual_tile、focus_fuser、tone 与 quality_gate_runner
（1,288–5,524 → 113–917 行，26 个生产子模块，最大 879 行）。应用层的 export/exif/image_stack/style_transfer、
image_processing/gpu_processing/lens_*/denoising、file_management/mask_generation/ai_processing/crop_analysis/
image_loader 与 crate 根 `lib.rs`（子模块在 `src/lib/`）也已拆分。N2 把测试支持、研究 oracle 和外置测试模块拆成主题文件，
测试名与测试总数不变。前端的大组件文件只移出顶层声明，`styles.css` 按原顺序拆成 `src/styles/app/` 下的 12 个文件。
约定、验证方法与仍超过 1000 行的文件见 [`source-layout.md`](source-layout.md)。

`src-tauri/src/panorama_utils/shared.rs` 保存 BRIEF descriptor、`ImageInfo`、`MatchInfo` 和 focus warp 等
跨阶段共享合同；`panorama_stitching.rs` 只做兼容重导出，低层模块直接依赖 `shared`。这一步不改变算法、
错误文本或输出格式。

`src-tauri/src/panorama_utils/stack_pipeline/` 当前包含 report、quality gate、focus fuser、virtual tile、
compositor、tone、residual warp、intra-station 和 test support 等模块。N0/N1 阶段只确认依赖方向和生产
入口，不自动接入未验证的 provider、oracle 或研究 solver。

`src-tauri/src/psd_loader.rs` 独立负责 PSD header、8/16-bit RGB/RGBA 像素解码、尺寸限制、取消检查和
错误分类；`image_loader.rs` 只按扩展名选择该解码器，避免把 Photoshop 格式判断和 zune-psd 细节继续
堆入通用图像入口。

### PSD 导入边界

PSD 当前按“一文件一张合成源图”导入：`src-tauri/src/psd_loader.rs` 使用 `zune-psd` 解码 PSD 的 composite image，转换为现有
`DynamicImage`/RGB 流程后再进入堆栈。不会把 PSD 内部图层展开成第二阶段的焦点帧，也不会改变 source
manifest、station 或 focus-layer 语义。首期支持 RGB/RGBA 的 8/16-bit PSD；CMYK、Lab、PSB、调整层和
智能对象等超出当前解码器能力的文件必须返回可诊断错误。透明度沿用普通非 RAW 导入的现有合成策略。

`/Users/dp/Downloads/书画拼图/马远` 登记为 PSD 回归集（14 张 PSD，单张约 8334×5528，16-bit RGBA），
用于验证批量导入、拼接和处处清晰输出。

PSD 组不会按 FocusStack 的单例站点 fallback 处理。入口检测到“全选输入均为 PSD 且请求为 FocusStack”时，
会自动改走 `Panorama + Auto` 的平面扫描路径：先用全组重叠关系求联合 photometric gain，再进入 seam
compositor；重叠区对 PSD 使用连续 detail blend，避免把每个文件直接写成一个硬 ownership 矩形。普通 RAW/JPEG
焦点堆栈和显式 Panorama 输入保持原有路径。

2026-10-28 的回归已验证 14/14 PSD 成功解码，匹配形成 13/13 连续邻接边；当前受控输出走联合
photometric + progressive seam，仍需用人工 ROI 对处处清晰和接缝连续性做最终视觉门禁，不能只用测试返回码
判定成片质量。该运行没有参考 ROI，Quality Gate 保持 `insufficient_evidence`。兰亭 45 张同日 1:1 回归仍让 45/45
源进入 compositor 并生成 `33895×13888` 成片，但现有已知画布尺寸差异仍使它与参考 Photoshop JPEG
无法字节一致；本次 PSD 导入没有改变这条既有边界。

## N1 拆分顺序

当前已完成 N1-1 的共享合同迁移：`shared.rs` 约 100 行，facade 保留旧路径，processing、mosaic、
stitching、compositor、test support 和 properties 已改为直接从 shared 取共享类型。N1-2a 已把
`Projection`、投影/逆投影、目标到源坐标映射、输出边界和画布尺寸计算移到
`panorama_utils/coordinates.rs`；`stitching.rs` 只保留兼容重导出，未改变调用路径或坐标含义。该模块有
Planar、曲面投影往返和带平移画布的 3 个单元测试，旧的 pixel-aligned facade 测试也继续通过。N1-2b 已把
区域 geometry/focus helper 移到 `panorama_utils/render_geometry.rs` 和 `panorama_utils/focus_geometry.rs`：
前者负责 transformed image/source bounds、focus output/render bounds，后者负责 mapped frame 与 shifted-mosaic
判定；两个模块均低于 1000 行，`stitching.rs` 仅保留兼容重导出，未改变阈值、坐标含义或回退行为。N1-3
已把 report schema、recorder 和原子 I/O 拆到 `report_schema_core.rs`、`report_schema_solver.rs`、
`report_schema_metrics.rs`、`report_document.rs`、`report_recorder.rs` 和 `report_io.rs`；各生产文件均低于
1000 行，`report.rs` 只保留兼容重导出和测试，未改变 `Stack_Report` 字段、schema 版本或写盘语义。N1-4 已
完成候选 pair 生成拆分：
`panorama_feature_pairs.rs` 为 239 行，`panorama_stitching.rs` 通过 facade import 继续提供原有内部调用名；
没有移动完整 feature matcher，也没有改变算法阈值、候选排序或报告语义。N1-5 已拆 focus feature preparation
和匹配合同；N1-6 又拆出候选、几何验证、focus rescue 和结果组装；N1-7 拆出 station relation gate、closure、
topology、pose seed、solver 和 Residual_Warp 证据；N1-8 拆出 focus grouping 与 capture groups；N1-9 把 `stitching.rs` 拆成 facade 与 34 个子模块；
N1-10 拆分其余超过 1000 行的生产文件（Virtual_Tile 关系匹配已在 `panorama_stitching/virtual_tile_*.rs`）。
下一步是把仍为单个超长函数的 `stitch_images_with_options` 与 `focus_tile_ownership_stitcher_with_finishing` 按阶段拆函数，
这需要单独的行为核对；
每一步都先跑定向测试，再跑完整库测试和兰亭保护性回归。

1. 共享类型、错误、报告和坐标转换。
2. 纯 geometry/focus helper、采样、mask 和颜色计算。
3. feature/matching、focus grouping、station solver、topology 和 closure。
4. focus warp、residual warp、ownership、seam 和 compositor。
5. virtual-tile store、tile source、BigTIFF/output adapter。
6. 顶层 prepare、执行流程、取消、缓存和最终发布。

每一步只移动代码和收敛依赖，不改算法阈值、坐标含义、回退行为或输出格式。除两个单函数文件外，新增的生产拆分模块
均不超过 1000 行；测试支持文件已在 N2 拆分。最终 facade 只保留稳定入口、阶段编排、常量和重导出。

## 验收边界

真实输入清单位于 [`tests/acceptance/focus-stack-datasets.json`](../tests/acceptance/focus-stack-datasets.json)。
兰亭 45 张 JPG 是强制保护基准，参考成片 `DSC08854_FocusStack.jpg` 只读。每次行为变更先运行数据契约，
再运行兰亭真实成片回归；输出和报告写入 `/private/tmp`，不得覆盖参考文件。2026-10-29 起清单共 11 个数据集：
`张好好2` 已从数据根目录移除并从清单删除，新增 `祁序1`（75 张 NEF）、`祁序2`（98 张 NEF）和 `马远2`（47 张 NEF），
三者暂无回归基线。

真实素材回归由 `npm run focus-stack-regression -- --dataset <id>` 执行（说明见
[`tests/acceptance/README.md`](../tests/acceptance/README.md)）。它调用 ignored harness
`real_focus_stack_fixture_from_env`，检查全部输入进入拼接顺序（streaming compositor 运行时还要求全部输入被
streaming）、per-pixel ownership 行、参考文件运行前后 digest 不变，并把 preview 与六个裁片的 digest 和
[`focus-stack-regression-baselines.json`](../tests/acceptance/focus-stack-regression-baselines.json) 对照。
test-only 模块 `panorama_acceptance_metrics.rs` 只测量成片：用 FAST/BRIEF + RANSAC 把输出配准到参考成片，
在内容 tile 上比较对比度归一化梯度能量 `mean |∇I|² / var I`，并分左/中/右三段统计；它不向生产渲染提供任何
参考像素或变换。digest 一致即证明职责迁移未改变成片；digest 变化时按 baseline 的清晰度下限判定，并仍需
人工查看 preview 和裁片。基线只保存计数、尺寸、digest 和指标摘要，不保存日志、报告、预览或临时路径。

2026-10-28 合并后的 `v1`（`8fa99498`）曾改变兰亭的执行路径：恢复连通性后 45 张形成 44 个 station（43 个单帧加
1 对近重复帧）、1 个连通分量，不再满足“全部单帧”的 source-scan fallback，改走 `layered_virtual_tile`。10% 回归输出 `2975×1283`（旧
legacy 基线为 `3388×1389`），45/45 进入拼接顺序，closure `converged`，tone `degraded`（最大边界
ΔE00 1.676 > 1.5），Quality Gate `insufficient_evidence`，因此 harness 的 layered native acceptance 在写出产物后
拒绝该报告；本机单次耗时约 25 分钟（旧 legacy 路径约 2.5 分钟）。与参考成片配准后（1,429 个 inlier）内容
tile 清晰度比值中位数 0.993、P05 0.827，左/中/右中位数 0.964/0.978/1.033，低于中位数一半的 tile 占 2.4%，
裁片笔画清晰、色调比旧 legacy 输出均匀。旧 legacy 基线在同一测量下为中位数 1.097、P05 0.809。该路径的 1:1
成片能力尚未确认：同日的 1:1 尝试运行约 95 分钟后只完成 258 个 Virtual_Tile relation 候选（10% 运行共记录 946 个），
进程峰值 footprint 9.3 GB，系统 swap 接近耗尽后被中止；旧 legacy 1:1 回归约 14 分钟完成。

`2a722f3d` 用 `capture_groups_form_singleton_scan`（`panorama_capture_groups.rs`）把 fallback 扩展到至少 9/10 的
station 为单帧的集合，兰亭因此回到 `legacy_single_layer_mosaic`。10% 回归 45/45 streaming、输出 `3381×1386`、
193 s，配准清晰度中位数 1.066、P05 0.846；1:1 回归 45/45 streaming、输出 `33821×13852`、每个像素都有 source
ownership，拼接 319 s 并写出 TIFF/app JPEG，配准清晰度中位数 1.111、P05 0.857、左/中/右 1.132/1.097/1.108，
100% 裁片笔画和印章纹理清晰。多帧 station 网格不受影响：Langyuan-10（4/3/3）和 Wenyuan-10（2/4/4）仍走 layered
路径，日志除耗时外与修复前一致；马远 PSD 输出逐字节不变。两组 10 张 NEF 在 10% 渲染下仍因 station 图断连而被
拒绝，这是缩小渲染的既有限制，验收需要 1:1。

以下各段记录合并前（`20c085c3` 及更早）的 legacy 基线，供追溯。当时已恢复兰亭 45/45 输入的完整渲染：N1-2a
后的 1:1 回归把 45 张都写入 streaming compositor，成片覆盖范围为
`33895×13888`，并成功写出 `/private/tmp/raw-editor-lanting-coordinates-full.tiff`（2.6 GiB）和同尺寸
`app.jpg`，预览为 `1600×656`；对应 10% 保护性回归输出为 `3388×1389`、预览 `1200×492`，整图和关键裁片
均保持笔画可辨。该恢复路径使用 `legacy_single_layer_mosaic` 的 source-scan fallback，避免移动扫描被误判为
多机位 station 后丢帧。当前输出可写盘但画布尺寸和参考 Photoshop 成片 `31065×11007` 不同；尺寸/裁切是否需要
进一步对齐是后续独立任务，不能用参考 JPEG 字节一致性替代“45 张完整且处处清晰”的验收。Stack_Report 仍记录
43 个 isolated、`DSC08871.jpg` native refinement 失败和 `quality_gate=insufficient_evidence`，这些是后续
station/quality 任务，不能反向解释为 source-scan 漏帧。

N1-2b 拆分后的 release 库测试为 `818 passed, 0 failed, 30 ignored`。当前 10% 兰亭回归仍为 45/45
source streaming，输出 `3388×1389`、预览 `1200×492`，日志位于
`/private/tmp/raw-editor-lanting-focus-geometry-10.log`，预览位于
`/private/tmp/raw-editor-lanting-focus-geometry-10.preview.jpg`；整图可读且未改变上述 Stack_Report 的诊断边界。
同一 release 二进制的 1:1 回归也保持 45/45 source streaming，输出 `33895×13888`，并写出
`/private/tmp/raw-editor-lanting-focus-geometry-full.tiff`（2.6 GiB）和同尺寸 app JPEG，预览位于
`/private/tmp/raw-editor-lanting-focus-geometry-full.preview.jpg`；这确认本轮纯 helper 移动没有丢失完整成片能力。

N1-3 report 拆分后的定向测试为 report `16 passed`、quality-gate `24 passed`；完整 release 库测试仍为
`818 passed, 0 failed, 30 ignored`。兰亭当前 10% 回归仍为 45/45 source streaming，输出 `3388×1389`，
并写出 `/private/tmp/raw-editor-lanting-report-split-10-stack-reports/stack-report-8ac825e8-46ba-4396-9d3f-1a45dc45131d.json`；
预览 `/private/tmp/raw-editor-lanting-report-split-10.preview.jpg` 保持完整可读。质量门仍按现有报告语义记录，
没有因为拆分把 `insufficient_evidence` 改写为通过。

N1-4 候选 pair 拆分后的定向测试为大堆栈邻域、自然文件名邻域、focus/panorama 中等规模候选和自然排序
共 `5 passed`；底层 `feature_provider`、`processing`、`registration` 测试分别为 `6`、`5`、`3 passed`。
完整 release 库测试仍为 `818 passed, 0 failed, 30 ignored`，数据集契约（含输入 hash）通过。兰亭 10% 保护
回归保持 45/45 输入、source streaming、输出 `3388×1389`、预览 `1200×492`，并写出报告
`/private/tmp/raw-editor-lanting-feature-pairs-10-stack-reports/stack-report-9716983f-75ac-4126-99a3-0697ac46a588.json`；
预览 `/private/tmp/raw-editor-lanting-feature-pairs-10.preview.jpg` 可读。报告仍按现有语义记录
`quality_gate=insufficient_evidence`，本次结构拆分未把它改写为通过。

N1-5 focus feature preparation/匹配合同拆分后的定向保护包括 feature alignment `4`、foreground/canonical/
scale 合同 `7`、guided overlap `2`、direct grid `9`、direct residual `3`、focus group `2`、polished matcher `1`
和 prior-free fallback `1`，均通过；N1-4 候选与 provider/processing/registration 定向测试也保持通过。
完整 release 库测试仍为 `818 passed, 0 failed, 30 ignored`，数据集契约（含输入 hash）通过。兰亭 acceptance
保持 45/45 source streaming，输出 `3388×1389`、预览 `1200×492`，报告写入
`/private/tmp/raw-editor-lanting-n15-stack-reports/stack-report-329977cb-d224-4c79-97e2-66f91d50d84b.json`；
预览 `/private/tmp/raw-editor-lanting-n15-10.preview.jpg` 整图可读。质量门仍为
`insufficient_evidence`，因为本轮只做职责迁移，没有虚构 ROI 证据或改变 rejection 原因。

N1-6 匹配执行体拆分后的定向保护测试全部通过：focus alignment `4`、mixed-focal `2`、adjacent same-focal
missing-EXIF `1`、guided overlap `2`、direct grid `9`、direct residual `3`、focus group `2`、polished matcher
`1`、prior-free fallback `1`、canonical direction `1`、canonical relation `1`、mirrored orientation `1`；
N1-4 的候选、feature provider、processing、registration 定向测试也全部通过。`cargo check --lib`、fmt、diff
和 `npm run focus-stack-quality:check` 均通过；release 全量库测试为 `818 passed, 0 failed, 30 ignored`，数据集
契约（含输入 hash）通过。
兰亭回归保持 45/45 输入，source streaming 输出 `3388×1389`、预览 `1200×492`，报告位于
`/private/tmp/raw-editor-lanting-n16-stack-reports/stack-report-b7c06bf3-5907-48e3-95b0-69bf521d5d1f.json`，
预览位于 `/private/tmp/raw-editor-lanting-n16-10.preview.jpg`；整图及 top-center、top/bottom left/right
裁片可读。报告仍记录 `selected_path=legacy_single_layer_mosaic` 和
`quality_gate=insufficient_evidence`，这表示回归保护通过而非 Quality Gate 已通过。

N1-7 station 拆分在合并后的 `v1` 上验证：`cargo check --lib --tests` 的警告集合与 `v1` 完全一致，fmt 通过，
release 全量库测试两次均为 `901 passed, 0 failed, 40 ignored`（含 3 个新 metrics 单测），closure/
station_relation/topology/spanning/station_pose/capture_group/residual 定向测试为 `23/11/11/1/3/9/30 passed`，
Node 契约和数据集契约（含输入 hash）通过。兰亭 10% 与马远 PSD 10% 回归的 preview 和六个裁片均与 `v1`
基线逐字节一致（`identical`）。验证中发现 `world_resampled_reference_uses_the_same_scale_and_still_detects_blur`
在 `v1` 上与 compositor 测试并行时约一半概率失败：前一个测试留下的进程级 tone run report 被该测试的
`source_roi` 回放。测试现在在取得 run scope 后先清空 tone 记录，同一组合 8 次运行 0 次失败（`v1` 为 4/8）。

N1-9 `stitching.rs` 拆分由两个并行计划在独立 worktree 中完成：计划 A 负责原文件前半区的画布、曝光、station
seam、全景接缝与分层合成；计划 B 负责后半区的 focus layer、颜色/tone、mask、对齐、融合、滤波、采样和测试
模块外置。每一步都要求 `cargo check --lib --tests` 的警告集合与基线一致。两个计划各自通过 release 全量库测试
`903 passed, 0 failed, 40 ignored`、Node 契约和兰亭/马远 PSD 10% 逐字节回归后，合并到 `f5b4e555`（PSD scan
mosaic 已进入 `v1`）之上。合并后的树：警告集合与 `v1` 一致，fmt/diff 检查通过；`v1` 的 `stitching.rs` 中 310 个
非 `use` 条目在 facade 或子模块中各出现一次，其中 305 个在归一化空白、可见性和路径限定后完全一致，3 个外置
测试模块与原模块体一致，2 个只差 rustfmt 的闭包花括号；release 全量库测试 `903 passed, 0 failed, 40 ignored`，
14 个 Node 契约通过；兰亭 10%（45/45 streaming，`3381×1386`）与马远 PSD 10%（`1554×1300`，`f5b4e555` 重新记录
的 PSD scan mosaic 基线）回归的 preview 和六个裁片均与 `v1` 基线逐字节一致。质量契约中
`current_seam_penalty … 32.0 … desired_owner` 的跨函数顺序断言因模块按文件名拼接而改为分别断言存在，仍保留
`32.0 → desired_owner` 的模块内顺序。

N1-10/N2 的长文件拆分由多个并行 track 在独立 worktree 中完成，每个 track 自行通过警告集合、条目级核对、release 全量
测试、Node 契约和兰亭/马远 PSD 10% 回归后，合并到同一个 integration 分支。合并期间还并入了另一个会话针对 `马远2`
的两项修复：排除帧在 Ownership_Map legend 中保留槽位但不拥有像素，分层 Virtual_Tile 载入原图时跳过这些槽位，
不再因缺少 source sampling 回退到旧渲染器；`Stack_Report` 中无可测重叠的 station relation 测量值（NaN 写成 `null`）
读回时恢复为 NaN，`rewrite_stack_report` 与验收 harness 不再报 `invalid type: null`。两项修复按拆分后的文件位置应用，
改动行与原提交逐行相同。合并后的完整树（含两项修复与数据集清单更新）：警告集合与基线一致，fmt 通过；release 全量库测试 `903 passed, 0 failed, 40 ignored`，943 个测试按名字与拆分前一一对应（531 个移入主题模块，路径对照见 [`test-path-map.tsv`](test-path-map.tsv)）；14 个 Node 契约通过，数据集契约含 `--hash-inputs`；兰亭 10%（45/45 streaming，`3381×1386`）与马远 PSD 10%（`1554×1300`）回归的 preview 和六个裁片与基线逐字节一致；前端 typecheck、lint、build、i18n:check 通过，构建出的 CSS 与拆分前逐字节相同。

N0 的基准行为门已通过，N1 当前推进到 N1-10；每一层拆分后必须重新跑数据契约、兰亭回归和完整 Rust 测试，
并保持上述 45/45 与清晰度门槛。
