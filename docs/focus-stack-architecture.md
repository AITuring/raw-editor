# 焦点堆栈代码架构

本文记录 2026-10-27 清理分支 `codex/layered-focus-modularize-2026-10-26` 的代码边界。
本轮提交 `222a4f6e` 完成了焦点堆栈顶层 `panorama_*.rs` 生产组件的拆分；它不表示整个
`src-tauri/src` 已经全部压到 1000 行以内。

## 当前分层

入口从 `panorama_stitching.rs` 进入，经过执行阶段、匹配、机位求解和 Virtual_Tile 合成，
最后调用 `panorama_utils` 的通用图像处理与质量门禁。顶层模块通过 facade 保留历史函数名和
测试访问路径，具体实现放在按职责命名的 sibling module 中。

| 层 | 当前职责 | 主要文件 |
| --- | --- | --- |
| 顶层 facade | 对外命令、稳定常量、模块重导出 | `panorama_stitching.rs` |
| 执行编排 | 缓存、入口、阶段顺序、准备/匹配/几何/求解/合成/收尾 | `panorama_execution*.rs`、`panorama_execution_phase_*.inc.rs` |
| 特征匹配 | 候选生成、特征对齐、成对匹配、模型拟合、原生 patch refinement | `panorama_feature_*.rs` |
| 焦点分组 | 捕获组、异常组件救援、顺序和全局优化 | `panorama_focus_grouping*.rs`、`panorama_focus_order.rs`、`panorama_focus_global_optimizer.rs` |
| 机位求解 | station grouping、relation gate、closure、station pose | `panorama_station_*.rs`、`panorama_focus_pose_*.rs` |
| focus warp | 水平/垂直物理边缘、局部 band、渲染辅助 | `panorama_focus_warp*.rs`、`panorama_focus_warp_render.rs` |
| Virtual_Tile solver | 类型、覆盖几何、引导/直接匹配、残差、金字塔、polish、prior、修复 | `panorama_virtual_tile_*.rs` |
| 共享适配 | 采集桥、图顺序、重叠几何、源准备、共享 helper | `panorama_capture_bridges.rs`、`panorama_graph_order.rs`、`panorama_overlap_geometry.rs`、`panorama_source_preparation.rs`、`panorama_shared_helpers.rs` |
| 通用 pipeline | compositor、mosaic、tone、quality gate、report、cache | `panorama_utils/stack_pipeline/`、`panorama_utils/stitching.rs`、`panorama_utils/mosaic.rs` |

执行阶段使用 `.inc.rs` 宏片段，是为了把必须共享局部状态的长流程分成可审阅单元；这些
片段不应重新把业务逻辑搬回顶层 facade。跨模块数据应优先通过已有 report/types/solver
接口传递，避免 sibling module 互相依赖实现细节。

## 行数边界

本轮已经达到的范围是：顶层 `panorama*.rs` 生产文件全部不超过 1000 行，最大的文件为
`panorama_focus_grouping.rs`（987 行），`panorama_stitching.rs` 为 973 行 facade。

以下内容仍未达到 1000 行，应作为后续拆分任务，而不是在文档中宣称“全仓库已拆完”：

| 文件 | 行数 | 后续拆分方向 |
| --- | ---: | --- |
| `panorama_utils/stitching.rs` | 9854 | compositor 入口、focus color/tone、ownership/seam、render adapter |
| `panorama_utils/mosaic.rs` | 5663 | ownership、photometric、sampling/refinement、diagnostics |
| `stack_pipeline/focus_fuser.rs` | 2640 | score、ownership graph、fallback/report |
| `stack_pipeline/quality_gate_runner.rs` | 2576 | ROI sampling、criteria、verdict/report |
| `stack_pipeline/virtual_tile.rs` | 2486 | cache store、lease、serialization、key/digest |
| `stack_pipeline/report.rs` | 2416 | schema、recorder、serialization、output digest |
| `stack_pipeline/tone.rs` | 2037 | samples、solve、field application、diagnostics |
| `stack_pipeline/quality_gate.rs` | 1450 | thresholds、metrics、criterion types |
| `stack_pipeline/intra_station.rs` | 1307 | analysis budget、control points、refinement |
| `stack_pipeline/degradation.rs` | 1277 | degradation policy、fallback classification、report adapter |
| `stack_pipeline/residual_warp.rs` | 1037 | residual field、identity/fallback、warp diagnostics |

测试支持文件也有超过 1000 行的情况（例如 alignment 4389 行、Virtual_Tile 3109 行、
acceptance 1766 行）。它们不计入本轮生产组件门槛，但应在后续按主题拆成测试模块，保持
每个测试文件能够独立定位失败原因。

## 验证基线

本分支已经验证：

- `cargo check --lib`；
- `cargo test --lib --no-run`；
- 完整 lib 测试：716 passed、0 failed、30 ignored；
- alignment 定向测试：126 passed；
- Virtual_Tile 定向测试：48 passed；
- `npm run focus-stack-quality:check`；
- `npm run typecheck`；
- `npm run build`。

这些结果证明拆分保持了当前行为和测试接口，但没有替代真实 RAW 数据的 Quality_Gate 验收。
真实素材门禁仍需单独记录输入 digest、accepted relations、closure、tone、ownership 和最终
质量结论。

## 维护规则

1. 新增生产模块先确定单一职责和调用方向，再决定是否放入 facade；facade 只保留稳定 API、常量和编排。
2. 单个生产文件接近 800 行时就开始拆分，1000 行是硬上限，不等到再次超过万行。
3. 模块拆分提交必须同时运行 `cargo fmt --check`、`git diff --check`、质量契约和相关定向测试。
4. 真实素材结果必须区分“管线连通/导出成功”和“Quality_Gate 通过”，不能用前者替代后者。
5. 任何改变 solver、ownership、tone 或 compositor 行为的修改，都要补充对应的确定性回归或真实素材诊断记录。

