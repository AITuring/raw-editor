# Focus Stack 当前架构与拆分边界

本文只描述当前工作区可见的生产入口、模块职责和后续拆分边界。模块拆分以
`.kiro/specs/layered-camera-group-focus-stitching/tasks.md` 的 N0–N9 为准；未接入生产的研究模块不算作
当前架构的一部分。

## 当前入口

`src-tauri/src/panorama_stitching.rs` 是 Rust 堆栈入口，负责输入预检、源图准备、特征匹配、全局关系、
机位求解、focus-stack 执行流程、取消、缓存和最终输出发布。当前仍是超大 facade 与实现混合文件，N1
完成前不把其中的实验入口视为稳定 API。

`src-tauri/src/panorama_utils/stitching.rs` 承担全局画布、focus warp、focus ownership、station fusion 和
顶层渲染回退；`mosaic.rs` 承担渐进接缝与 detail-preserving mosaic；`processing.rs` 承担通用采样、滤波、
几何和颜色 primitive。它们仍有跨职责实现，不能据此宣称已经完成模块化。

`src-tauri/src/panorama_utils/shared.rs` 保存 BRIEF descriptor、`ImageInfo`、`MatchInfo` 和 focus warp 等
跨阶段共享合同；`panorama_stitching.rs` 只做兼容重导出，低层模块直接依赖 `shared`。这一步不改变算法、
错误文本或输出格式。

`src-tauri/src/panorama_utils/stack_pipeline/` 当前包含 report、quality gate、focus fuser、virtual tile、
compositor、tone、residual warp、intra-station 和 test support 等模块。N0/N1 阶段只确认依赖方向和生产
入口，不自动接入未验证的 provider、oracle 或研究 solver。

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
1000 行，`report.rs` 只保留兼容重导出和测试，未改变 `Stack_Report` 字段、schema 版本或写盘语义。下一步按
feature/matching、station solver、compositor 的顺序推进；
每一步都先跑定向测试，再跑完整库测试和兰亭保护性回归。

1. 共享类型、错误、报告和坐标转换。
2. 纯 geometry/focus helper、采样、mask 和颜色计算。
3. feature/matching、focus grouping、station solver、topology 和 closure。
4. focus warp、residual warp、ownership、seam 和 compositor。
5. virtual-tile store、tile source、BigTIFF/output adapter。
6. 顶层 prepare、执行流程、取消、缓存和最终发布。

每一步只移动代码和收敛依赖，不改算法阈值、坐标含义、回退行为或输出格式。拆分后的生产文件和测试
支持文件都必须不超过 1000 行；facade 只保留稳定入口、阶段编排、常量和重导出。

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

N0 的基准行为门已通过，可以开始 N1 的职责拆分；每一层拆分后必须重新跑数据契约、兰亭回归和完整 Rust 测试，
并保持上述 45/45 与清晰度门槛。
