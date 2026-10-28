# Focus Stack 当前架构与拆分边界

本文只描述当前工作区可见的生产入口、模块职责和后续拆分边界。模块拆分以
`.kiro/specs/layered-camera-group-focus-stitching/tasks.md` 的 N0–N9 为准；未接入生产的研究模块不算作
当前架构的一部分。

本文按 `v1` 分支 2026-10-28 的 N1-6 代码基线修订。新增匹配模块的行数以当前工作区为准；历史记录中的
行号和临时报告路径只用于追溯，不能作为当前模块定位或验收依据。

## 当前入口

`src-tauri/src/panorama_stitching.rs` 是 Rust 堆栈入口，负责输入预检、源图准备、特征匹配、全局关系、
机位求解、focus-stack 执行流程、取消、缓存和最终输出发布。当前仍是超大 facade 与实现混合文件，N1
完成前不把其中的实验入口视为稳定 API。

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

`src-tauri/src/panorama_utils/stitching.rs` 承担全局画布、focus warp、focus ownership、station fusion 和
顶层渲染回退；`mosaic.rs` 承担渐进接缝与 detail-preserving mosaic；`processing.rs` 承担通用采样、滤波、
几何和颜色 primitive。它们仍有跨职责实现，不能据此宣称已经完成模块化。

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
和匹配合同；N1-6 又拆出候选、几何验证、focus rescue 和结果组装，下一步进入 station solver、topology 和
closure 的职责拆分；
每一步都先跑定向测试，再跑完整库测试和兰亭保护性回归。

1. 共享类型、错误、报告和坐标转换。
2. 纯 geometry/focus helper、采样、mask 和颜色计算。
3. feature/matching、focus grouping、station solver、topology 和 closure。
4. focus warp、residual warp、ownership、seam 和 compositor。
5. virtual-tile store、tile source、BigTIFF/output adapter。
6. 顶层 prepare、执行流程、取消、缓存和最终发布。

每一步只移动代码和收敛依赖，不改算法阈值、坐标含义、回退行为或输出格式。本轮新增的生产拆分模块均不
超过 1000 行；测试支持文件的进一步拆分列入 N2。最终 facade 只保留稳定入口、阶段编排、常量和重导出。

## 验收边界

真实输入清单位于 [`tests/acceptance/focus-stack-datasets.json`](../tests/acceptance/focus-stack-datasets.json)。
兰亭 45 张 JPG 是强制保护基准，参考成片 `DSC08854_FocusStack.jpg` 只读。每次行为变更先运行数据契约，
再运行兰亭真实成片回归；输出和报告写入 `/private/tmp`，不得覆盖参考文件。

当前已恢复兰亭 45/45 输入的完整渲染：N1-2a 后的 1:1 回归把 45 张都写入 streaming compositor，成片覆盖范围为
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

N0 的基准行为门已通过，N1 当前推进到 N1-6；每一层拆分后必须重新跑数据契约、兰亭回归和完整 Rust 测试，
并保持上述 45/45 与清晰度门槛。
