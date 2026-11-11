> 历史快照：2026-11-04 文档整合前的任务原文，含83个勾选条目。下文“唯一执行计划”“当前”等措辞只代表当时记录，不再维护。当前状态与推进顺序统一见[项目进度与任务总览](../project-progress.md)。

# 堆栈模块开发任务（新基线）

## 使用规则

本文件是后续堆栈模块开发的唯一执行计划。旧版任务编号、阶段表、逐轮实验记录、临时报告路径和历史
分支状态不再作为任务状态，也不作为验收依据。历史报告可以留在 Git 历史和 `.agents/` 归档中，但不能
直接勾选本文件中的任务。

当前基线是 `v1`：2026-10-28 将远端 `20c085c3`（包含 `d58ff8aa` 基线及其后续更新）与本地快照 `a4f27d09` 合并。
本次保留的候选修改仍须经过本文件的 N0 分流和验证，合并、编译和回归测试不代表真实素材验收完成。
合并前的本地执行记录保存在 `docs/focus-stack-local-work-history.md`，仅供历史追溯，不改变下列任务状态。

`.worktrees/layered-focus-cleanup` 的模块化分支不是当前基线；其中的行数和完成状态不能直接写入本文件，
只有经过 N0 的逐项迁移、编译和回归后才能合并进来。

执行纪律：

- 先记录证据，再改变行为；所有行为变更都要有 baseline、输入 digest、输出 digest 和可复现命令。
- 每个阶段只解决一个职责层，阶段结束后再进入下一层；不得把测试迁移、算法调参和输出改造混在一个提交中。
- 生产代码、测试支持代码和 test-only oracle 都必须保持职责清晰；单个文件最长 1000 行。
- 原有门槛保持不变，尤其是 station relation 的尺度、空间支持、方向、双向匹配和闭环门槛；没有独立证据
  不得放宽阈值、强制连接机位或把外部变换写入生产关系图。
- Quality Gate 在真实素材连续通过前保持 record-only；导出成功、结构测试通过或 `not_run` 都不能代替门禁通过。

## 当前代码基线

### 2026-11-03 已验收范围与集成收口

当前源码身份与修改归属见 `docs/focus-stack-accepted-baseline.md`，逐文件清单见
`docs/benchmarks/accepted-baseline-inventory-2026-11-03.json`。v28 固定基线包含 497 个 Rust 源码文件；
v31 冻结源码为 499 个文件（相对 v28 为 6 个既有文件修改、2 个新增文件）；v32 当前为 500 个文件，
相对 v31 修改 9 个既有文件、增加 1 个测试模块，另行冻结二进制和清单。
既有 9 张连续 NEF、scale=1 冷/热验收按用户确认的原性能目标和既有门禁收口。
该状态只补充本节所列范围，不覆盖下文其他素材的失败或待验收状态。

- [x] P0 路径观测、singleton-dominated scan 修复及运行统计。
- [x] 统一 SourcePreflight、受限 source/ROI/native fusion 缓存及 render→Quality Gate 来源复用。
- [x] station owner 证据、criterion-specific ROI、native closure/local-scale 修复及来源追溯。
- [x] 同一 9 张 NEF 的原生 TIFF 冷/热验收：QG pass / 8 ROI，输出 SHA 一致，覆盖区域 owner 缺失为 0。
- [x] 局部亚像素配准和多区域稳定性验证；未证明候选稳定胜出，保持既有 owner 选择规则。
- [x] 当前工作区 macOS app 构建及单张 NEF 导入、曝光调整、完整分辨率 JPEG 导出冒烟；
  独立副本 8256×5504、sRGB，原始素材未修改。该项不扩展为全部桌面交互或跨平台发布通过。
- [ ] 默认小缓存预算、其他真实素材和完整桌面交互的泛化验收；内容不适用、degraded 和旧诊断失败继续保留。

当前推进顺序（2026-11-03 默认预算与跨素材验收）：

- [x] 同一 9 张 NEF、scale=1，清除预算覆盖值，按 source 512MiB / native fusion 1GiB 的实际默认配置
  完成串行冷/热运行；比较 TIFF、既有 Quality Gate、来源/几何证据、解码、fusion、时间及 RSS。
  质量/预算是否通过与性能是否改善分开记录，不把热跑等同于跨进程缓存命中。
  实测 QG 均 pass / 8 ROI，TIFF 与 v28 完全相同，冷/热业务一致；耗时 1724.400s / 1262.314s，
  两次 RAW 解码均 176、fusion 均 21，两级缓存均 0 hits。仅完成默认配置的质量/预算验收，
  不把默认预算下的性能优化标记为完成。摘要：`docs/benchmarks/sleeping-9-v29-default-cache-acceptance.json`。
- [x] 修复默认预算下的有效复用并完成同组冷/热复验（v31）：大对象匿名无损后备、完整证据恢复、
  SHA 校验和失败回退；RAM 仍为 source 512MiB / native 1GiB，新增临时磁盘上限 8GiB / 4GiB。
  冷/热 418.756s / 451.682s，RAW 18 / 18、fusion 6 / 6；QG pass / 8 ROI、station owner evidence 8，
  TIFF 与 v28/v29 相同。RSS 为 8.15 / 8.85GiB，较 v29 略升但通过原门限；后备峰值 4.63 / 2.70GiB。
  定向 50 passed / 2 ignored，全库 1021 passed / 40 ignored，均 0 failed；格式、生产 check 与契约通过。
  摘要：`docs/benchmarks/sleeping-9-v31-bounded-backing-acceptance.json`。该项不替代下文跨素材验收。
- [x] 默认预算冷/热完成后，依次执行 Langyuan-10、Wenyuan-10 的原生入口和兰亭 45 张行为保护回归，
  完成运行与证据归档（通过 0/3）：阆苑/文苑为 3/2 分量断连、QG not_run；兰亭成功导出
  33821×13852 TIFF，但 coverage alpha 合同失败、QG insufficient_evidence，外层验收均退出 1。
  摘要：`docs/benchmarks/real-materials-v30-default-cache-acceptance.json`；不升级到 84/126 张。
- [x] N4 坐标与诊断子项（v32）：source seed 改用 station-local geometry，保留全部候选尝试并区分
  拟合与过门禁；定向 3 passed，全库 1024 passed / 40 ignored。两套真实 10 张子集复验恢复
  16/16 次 source 拟合，但全部选中假设仍被 scale 拒绝；连通分量仍为 3/2、QG not_run、无 TIFF。
  文苑已接受关系与两套原生平面 SHA 不变。该子项完成不等于下项 N4 几何验收完成。
  同组 9 张默认冷/热回归均 pass / 8 ROI，TIFF 与 v28/v31 相同，RAW 18/18、fusion 6/6，
  native wall 395.372s / 392.813s；原有 12 个证据部分与 v31 一致。
  摘要：`docs/benchmarks/cross-station-v32-local-source-seed-acceptance.json`。
- [x] N4 当前 10 张子集尺度根因审计（v33）：原生倍率、像素中心映射、anchor/crop 与完整尺度门禁
  适用范围已核对；新增只读点距/三角形、四区域、候选 anchor 和光学回投审计，7 项测试通过。
  独立 RAW anchor probe 与完整子集复跑均完成；后者仍为 3/2 分量、0/1 accepted、QG not_run。
  生产 Rust 0 改动，9 张 v32 基线的二进制、源码、报告和 TIFF SHA 已复核；不是新的冷/热运行。
  摘要：`docs/benchmarks/cross-station-v33-scale-root-cause.json`，解释边界见性能文档第 34 节。
- [x] N4 v34 方案与冻结准备：按本轮授权明确跨机位真实倍率、局部形变、输出采样和画质门禁的边界；
  方案见 `docs/focus-stack-variable-magnification.md`，新增需求 6.11，保留旧范围兼容与修复增量限制。
- [x] N4 v34 有界倍率实现与两套10张复验：候选/solver共用空间留出、四区域、Jacobian证据；
  定向4项、全库1028 passed /40 ignored、3项Node contract与7项审计测试通过。两套连通均恢复1分量，
  各9张匹配平面SHA与v32相同。阆苑QG fail/8ROI（覆盖97.19%<98%），文苑闭环P95 16.23px、
  输出换算19.13px，QG not_run/0ROI；均外层exit1，未标为完整验收通过。
- [x] N4 v34 9张默认缓存冷/热正对照：均pass/8ROI、RAW18/18、fusion6/6，wall401.591/376.232s，
  RSS9.707/9.513GB；TIFF与冻结基线相同，12个非计时证据部分均与v32一致，默认预算和来源证据保留。
  阆苑已做预览/4个原生裁切观察，边界缺口与QG失败保留；文苑无输出不标视觉通过。
  摘要：`docs/benchmarks/cross-station-v34-variable-scale-acceptance.json`。
- [x] N4 v35 文苑联合原生几何与完整真实验收：联合优化前移到剪边前，分离几何修正和输出密度；
  真实10张保留3条关系，原生P95由16.2335降至1.81074px，输出P95=2.13611px、density1.179688。
  closure converged，物理覆盖99.472410%，QG pass/8ROI/8 station owner evidence，原生验收exit0。
- [x] N4 v35 有效来源域：完整原生逆映射独立判定支持域，alpha仅参与分子，98%门槛不变；
  保留旧Composition计数与新域审计，3项定向测试通过，两套最终TIFF与测量修正前逐字节相同。
- [x] N4 v35 阆苑覆盖、参考配准及原生视觉复验：覆盖99.441301%、QG pass/8ROI；
  最终参考fit99/237、held114/211、P95=1.910355px（原整幅参考2048格），验证域27.3995%，
  仅域内52个内容tile比较，不外推、不将PS像素用于渲染；3项参考测试通过，局部视觉边界保留。
  此项完成不等于完整原生验收：最终exit101/外层1，原因是closure=no_constraints。
- [x] N4 v35 同一9张默认缓存冷/热质量与缓存回归：QG pass/8ROI/8 owner evidence，RAW18/fusion6，
  source/native hits113/15，TIFF与v28逐字节一致；11个非QG证据部分与v34不变，旧基线/29个输入保护核验通过。
  wall532.148/677.405s、RSS8,794,636,288/9,421,586,432B；不将此勾选解释为性能无回退。
- [ ] v35 计时回退归因：冷/热wall比v34增加32.51%/80.05%，精确覆盖约2秒不能解释全部差异；
  后续用冻结v34/v35同机受控交错重复运行，分离融合、IO与系统负载，保留本次慢跑原始记录。
- [ ] N4 阆苑独立闭环证据：当前只有0–1/1–2两条有效边；0–2候选未通过原生证据门禁。
  先查真实重叠和独立关系，禁止以树图零字段冒充实测闭环或改记converged；保持现有拒绝。
- [ ] N4 跨素材完整验收：几何连通不替代闭环、画质和原生视觉验收；所有失败阶段据实保留，
  不扩大到84/126张，不覆盖9张NEF正对照或未提交修改。
- [ ] N6/N7 legacy 输出与证据：兰亭补齐 Coverage/alpha 导出合同，处理可见边界/色调问题，
  增加可追溯的画质证据；45/45 流式处理和通用 TIFF 检查通过不能替代该项验收。

### 需要纳入 N0 审计的生产范围

- 顶层编排：`src-tauri/src/panorama_stitching.rs`。
- 拼接核心：`src-tauri/src/panorama_utils/stitching.rs`、`mosaic.rs`、`processing.rs`。
- 堆栈阶段：`stack_pipeline/{focus_fuser,quality_gate_runner,virtual_tile,report,tone,quality_gate,
  intra_station,degradation,residual_warp,compositor}.rs`。
- 测试与属性支持：`panorama_stitching_*_tests.rs`、`panorama_virtual_tile_tests.rs`、
  `stack_pipeline/{properties,test_support}.rs`。

当前工作区仍有超过 1000 行的历史生产或测试实现；本轮新增的 N1-6 生产组件均已低于 1000 行。N1/N2
继续按实际文件重新测量，不引用其他分支的行数结论；测试支持和属性测试的拆分留在 N2。

### 未提交修改的分流对象

N0 要逐项登记以下候选修改的来源、依赖、测试和生产状态：

- 可保留候选：canonical station relation gate、decoded/preflight、cache/proxy 坐标契约、确定性辅助、
  Quality Gate rejection/effective-pixel 记录和 BigTIFF 路由；每项仍须通过 N0 的 clean regression。
- 需要重新验证后才能接入生产：`feature_provider`、`focus_pyramid`、`proxy`、`station_seam`、
  `tile_source`、Residual_Warp 接线和新的 compositor 路径；owner smoothness、FlatArtwork、direct
  matcher 和 global/prior-free bridge 先隔离到实验路径。
- 只允许 test-only 或研究隔离：`panorama_structural_oracle.rs`、`owner_quality_oracle.rs`、
  `final_world_owner_oracle.rs` 以及任何 reduced score、ROI reservation、QG replay、外部 Source_RAW
  关系或未经 holdout 的 pose/warp oracle。
- `src/components/modals/BatchGeometryModal.tsx` 属于无关 UI 修改，不纳入堆栈基线。
- 原始 command log、Stack Report、preview、RAW 和绝对临时目录不进入源码提交；只保留脱敏后的 schema、
  计数和可复现命令。

### 必须保持的产品契约

- Rust 是默认生产链，负责 RAW、全分辨率 IO、几何验证、融合、输出、取消、内存和确定性；Python/OpenCV
  只能作为可选代理 provider 或离线对照，失败必须回退 Rust 并写入报告。
- Source_RAW 的 digest、来源 provenance、Coverage_Mask、Ownership_Map 和透明区域语义必须贯穿渲染、
  诊断和输出；不得用 Photoshop 结果、文件名顺序或外部变换替代生产证据。
- station relation 必须经过同一 typed canonical view 和最终共享 gate；proposal 只能产生搜索种子，
  不能直接升级为 accepted relation。
- owner 证据必须来自当前运行的最终坐标和 Source_RAW 采样；test-only reduced oracle 不能替代正式 owner。
- 没有足够双向闭环证据时 Residual_Warp 保持 identity，并把原因写入报告。

### 当前验收基线

- 真实素材回归统一用 `RAW_EDITOR_FOCUS_STACK_DATA_ROOT=/Users/dp/Downloads/书画拼图 npm run
  focus-stack-regression -- --dataset <id>`；产物写入 `/private/tmp/raw-editor-regression`，基线（计数、尺寸、
  preview/裁片 digest、配准清晰度摘要）记录在 `tests/acceptance/focus-stack-regression-baselines.json`。
- 合并后的 `v1`（`8fa99498`）曾把兰亭改走 `layered_virtual_tile`：45 张形成 44 个 station（43 个单帧加 1 对近
  重复帧）、1 个连通分量，不再满足“全部单帧”的 source-scan fallback。10% 回归输出 `2975×1283`、约 25 分钟，tone
  `degraded`（最大边界 ΔE00 1.676 > 1.5）、Quality Gate `insufficient_evidence`，layered native acceptance 拒绝该
  报告；1:1 尝试约 95 分钟只完成 258/946 个 Virtual_Tile relation 候选，峰值 footprint 9.3 GB，被中止。
- `2a722f3d` 把 fallback 扩展到 singleton-dominated station 集合（至少 9/10 的 station 为单帧），兰亭回到
  `legacy_single_layer_mosaic`：10% 回归 45/45 streaming、`3381×1386`、193 s；1:1 回归 45/45 streaming、
  `33821×13852`、每个输出像素都有 source ownership，拼接 319 s 并写出 TIFF/app JPEG。配准参考成片后 1:1 内容 tile
  清晰度比值中位数 1.111、P05 0.857，左/中/右 1.132/1.097/1.108，低对比 tile 0.1%；100% 裁片笔画和印章纹理清晰。
  Langyuan-10（station 4/3/3）与 Wenyuan-10（2/4/4）仍走 layered 路径，日志除耗时外与修复前一致（10% 渲染下仍因
  station 图断连而拒绝，属既有限制）；马远 PSD 10% 输出逐字节不变。
- 以下兰亭条目是合并前（`20c085c3` 及更早）legacy 路径的历史基线。
- 兰亭 45 张基准已恢复到 45/45 输入完整参与渲染。N1-2a 后的 1:1 回归输出 `33895×13888`，成功写出
  `/private/tmp/raw-editor-lanting-coordinates-full.tiff` 和同尺寸 app JPEG，预览 `1600×656`；日志包含 45 条
  source streaming 记录，整图及关键裁片笔画可辨，参考文件保持只读。
- 兰亭当前通过 source-scan fallback 走 `legacy_single_layer_mosaic`，避免把移动扫描误判成多机位 station 后
  在 LayeredVirtualTile 中排除帧。Stack_Report 仍会保留 grouping isolated 的诊断事实；这不等于 geometry/quality
  gate 已通过，后续 N4 需要单独解决 station grouping 与画布裁切语义。
- N1-2b 后的 10% 保护性回归仍为 45/45 source streaming，画布 `3388×1389`，预览 `1200×492`；报告仍保留
  `DSC08871.jpg` native refinement 失败、43 个 isolated 和 `quality_gate=insufficient_evidence`，因此只确认合成
  行为没有回退，不把本次 preview-only 运行标记为 Quality Gate 或正式输出通过。
- 当前恢复保住了“45 张完整且处处清晰”的行为门，但输出画布尚未与参考 Photoshop 成片 `31065×11007` 对齐；
  Stack_Report 的 `output.written=false` 是因为验收测试在报告生成后另行写出 TIFF/app JPEG，不能据此判断测试文件未写出。
- 数据契约覆盖 9 个数据集并通过：兰亭 45、马远 PSD 14、阆苑女仙 84、文苑图 126、张好好2 244、睡觉
  45、枫鹰雉鸡 63，以及头部 17、底部 24。2026-10-29 起为 11 个：张好好2 已移除，新增祁序1 75、祁序2 98、
  马远2 47（NEF，暂无回归基线）；`--hash-inputs` 全部通过。
- Rust 完整 release 库测试在 N1-2b 后为 `818 passed, 0 failed, 30 ignored`。两个属性测试的回归根因是已有 station anchor
  被重新按清晰度排序；现在已改为“已有记录锚点优先、无记录时才按清晰度选锚点”，属性测试和完整套件均通过。
- Langyuan/Wenyuan 的 owner、effective-pixel、boundary、tone 和 residual evidence 仍不能视为通过；不能用兰亭
  预览恢复代替完整门禁。owner smoothness/hard terminal、ROI reservation、QG replay、Option B、spatial tone、
  直接 Source_RAW bridge 和 structural oracle 继续保持 test-only/research 隔离。

N3 继续用固定 manifest 重新生成每套素材的解码、机位图、Quality Gate 和输出证据；兰亭的 45/45 行为门是进入
N1 后每个职责层的回归保护条件，其他数据集的历史结果不能直接标记为当前通过。

## N0：冻结新基线并分流未提交修改

- [x] 保存当前 `git status`、tracked diff、untracked 清单和忽略规则摘要；不 reset、stash、clean 或覆盖
  工作区已有代码。2026-11-03 保存 237 个展开文件的哈希与内容快照，见基线文档。
- [x] 建立 `tests/acceptance/focus-stack-datasets.json` 和 `focus-stack-datasets:check`；登记兰亭、马远 PSD、
  阆苑女仙、文苑图、张好好2、睡觉、枫鹰雉鸡及其头部/底部输入，排除派生图；兰亭 45 张输入的参考
  成片路径、SHA-256、尺寸和文件大小已冻结。2026-10-29：张好好2 移除，新增祁序1、祁序2、马远2。
- [x] 恢复兰亭 45 张完整成片行为：45/45 输入进入 streaming compositor，1:1 回归成功写出 TIFF/app JPEG，
  整图与左中右裁片保持清晰；参考成片未覆盖。输出尺寸/裁切对齐仍作为后续独立门禁。
- [x] `v1` 合并后重新确认兰亭行为门：singleton-dominated scan 回到 legacy compositor（`2a722f3d`），10% 与 1:1
  回归及 baseline 见“当前验收基线”。
- [x] 为每个修改建立 `keep`、`adapt`、`test-only`、`exclude` 四类登记，注明生产入口、测试入口和证据。
  2026-11-03 完整登记开始时 237 个文件；exclude 仅表示不进入堆栈基线，原文件保留。
- [ ] 清理源码树中的 raw/preview/命令报告引用；历史 artifact 只保留在本地归档，不进入生产提交。
- [x] 在不改变行为的前提下完成 `cargo fmt --check`、`cargo check --lib`、定向 Rust 测试、Node 契约、
  `npm run typecheck`、`npm run build` 和 `git diff --check`。2026-11-03 全量 Rust 1017/0/40；
  同步报告 `performance` 契约和 12 语言缺失键后 Node、类型、lint、翻译检查通过。
  打包固定工具链 Rust 1.96.1 的完整库测试也为 1017/0/40，macOS app 与单图桌面冒烟通过。
  全库 Prettier 的 48 个既存格式告警单独保留，不算作已通过。
- [x] 将筛选后的当前代码形成一个可回滚基线提交。2026-11-03 使用独立本地分支
  `codex/accepted-stack-v28-baseline` 冻结已验收源码，当前 `v1`、index 和其他未提交修改保持不动。
  本次是 N1 拆分后性能阶段的补充基线，不追溯改写旧阶段的执行顺序。

**N0 通过条件**：每个未提交文件都有归属；没有未登记的实验 oracle 或大文件 artifact；基线能编译，且
所有失败都能归因到当前代码而不是混杂的历史工作区。兰亭参考文件不得被测试覆盖，且每次基线/行为变更
后都必须重新确认其 SHA-256、尺寸和文件大小未改变。

**N0 当前状态（2026-11-03）**：已完成当前未提交内容的快照、逐文件分流、已验收堆栈基线固定和
规定的编译/契约/库测试。研究诊断、其他前端工作和原始产物保持隔离；源码引用清理项仍未完成，
不把 N0 或整个发布门禁标记为全部完成。本轮真实素材范围是前述 9 张 NEF，其他素材不追溯改判。

**N0 历史兰亭保护记录**：45 张全部进入顺序并完成 streaming compositor，1:1
成片范围为 `33895×13888`，预览 `1600×656`，并已在 `/private/tmp` 写出 TIFF/app JPEG，整图和关键裁片保持
可读；参考成片仍为只读且 SHA-256、尺寸、文件大小未改动。输出尺寸/裁切与参考不同，Quality Gate 和最终发布
仍未宣称通过；N1 可以在保持该行为门的前提下继续，尺寸对齐另列为后续任务。

## N1：按职责拆分生产代码

拆分只移动代码和收敛依赖，不改变算法阈值、坐标含义、回退行为或输出格式。推荐顺序：

1. 共享类型、错误、报告和坐标转换。
2. 纯 geometry/focus helper、采样、mask 和颜色计算。
3. feature/matching、focus grouping、station solver、topology 和 closure。
4. focus warp、residual warp、ownership、seam 和 compositor。
5. virtual-tile store、tile source、BigTIFF/output adapter。
6. 顶层 prepare、执行流程、取消、缓存和最终发布。

- [x] N1-1 共享类型迁移：新增 `panorama_utils/shared.rs`，承载 BRIEF descriptor、`ImageInfo`、
  `MatchInfo` 和 focus warp 合同；facade 保留兼容重导出，低层模块改为直接依赖 shared，编译和定向测试通过。
- [x] N1-2a 纯坐标契约迁移：新增 `panorama_utils/coordinates.rs`，承载 `Projection`、投影/逆投影、
  `map_target_to_source`、`output_bounds`、`pixel_aligned_canvas` 和 `output_canvas_dimensions`；
  `stitching.rs` 保留兼容重导出，3 个新坐标单测、旧 facade 单测、property 54、数据集契约、兰亭 10% 回归和
  1:1 写盘回归通过。
- [x] N1-2b 纯 geometry/focus helper 迁移：新增 `panorama_utils/render_geometry.rs` 和
  `panorama_utils/focus_geometry.rs`，分别承载 transformed image/source bounds、focus output/render bounds、
  mapped frame 与 shifted-mosaic 判定；模块均低于 1000 行，保留 `stitching.rs` 兼容重导出。新增 geometry
  单测、shifted-mosaic 定向测试、virtual-tile/focus-analysis 回归、数据集契约、兰亭 10% 回归和完整 release
  库测试通过，行为和诊断阈值未改变；随后用同一 release 二进制完成兰亭 1:1 写盘回归，输出
  `33895×13888`，成功写出 `/private/tmp/raw-editor-lanting-focus-geometry-full.tiff`、同尺寸 app JPEG
  和预览，45/45 source streaming 保持不变。
- [x] N1-3 report schema/recorder 拆分：新增 `report_schema_core.rs`、`report_schema_solver.rs`、
  `report_schema_metrics.rs`、`report_document.rs`、`report_recorder.rs` 和 `report_io.rs`；各生产文件均低于
  1000 行，`report.rs` 保留兼容 facade、重导出和测试，未改变 `Stack_Report` schema 版本、字段、阈值或原子
  写盘语义。report 定向测试 `16 passed`、quality-gate 定向测试 `24 passed`、完整 release 库测试
  `818 passed, 0 failed, 30 ignored`，数据集契约通过；兰亭 10% 回归保持 45/45、`3388×1389`，报告成功写出。
- [x] N1-4 候选 pair 生成拆分：新增 `panorama_feature_pairs.rs`（239 行），承载输入/自然文件名排序、邻域
  候选、focus/panorama 候选策略、受限视觉检索和大堆栈候选扩展；`panorama_stitching.rs` 只保留 facade import
  和后续编排调用，未改变阈值、排序、回退或报告语义。候选定向测试 `5 passed`，底层
  `feature_provider`/`processing`/`registration` 为 `6/5/3 passed`，完整 release 库测试
  `818 passed, 0 failed, 30 ignored`，数据集契约（含输入 hash）通过；兰亭 10% 回归保持 45/45、
  `3388×1389`、预览 `1200×492`，报告写入 `/private/tmp/raw-editor-lanting-feature-pairs-10-stack-reports/`。
- [x] N1-5 focus feature preparation 与匹配合同拆分：新增 `panorama_focus_feature_prep.rs`（279 行）和
  `panorama_match_contract.rs`（262 行），分别承载 bounded focus 特征、foreground range/mask、top feature、
  foreground predicate，以及 canonical relation 方向、空间支持、homography/station scale 和 adjacent
  same-focal pair 合同；完整 `match_image_pair_with_seed` 暂留 facade，避免职责迁移混入算法变化。feature/
  contract、guided/direct/polished/prior-free、候选和底层 provider/processing/registration 定向测试均通过，
  完整 release 库测试 `818 passed, 0 failed, 30 ignored`，数据集契约（含输入 hash）通过；兰亭 acceptance
  保持 45/45、`3388×1389`、预览 `1200×492`，报告写入 `/private/tmp/raw-editor-lanting-n15-stack-reports/`，
  `quality_gate=insufficient_evidence` 语义未改变。
- [x] N1-6 匹配执行体拆分：将 `match_image_pair_with_seed` 按候选生成、几何验证、focus rescue 和结果组装拆成
  单一职责模块：`panorama_match_candidates.rs`（97 行）、`panorama_match_geometry.rs`（584 行）、
  `panorama_match_focus_rescue.rs`（270 行）和 `panorama_match_result.rs`（28 行）。入口保留原有
  `match_image_pair` 合同与阶段顺序，未改变阈值、坐标、回退或报告语义；新增模块均低于 1000 行。
  focus/mixed-focal/canonical/direct-grid/direct-residual/polished/prior-free 定向测试、N1-4 底层定向测试、
  `cargo check --lib`、fmt/diff 检查和 `npm run focus-stack-quality:check` 均通过；release 全量库测试
  `818 passed, 0 failed, 30 ignored`，数据集契约（含输入 hash）通过。兰亭 acceptance 保持 45/45 source
  streaming，输出 `3388×1389`、预览 `1200×492`，
  报告写入 `/private/tmp/raw-editor-lanting-n16-stack-reports/`，`quality_gate=insufficient_evidence` 保持原语义。
- [x] N1-7 station solver/topology/closure 拆分：在合并后的 `v1` 上新增九个单一职责模块——
  `panorama_station_relation_gate.rs`（566 行，证据类型、requirement-6 测量与共享 gate、overlap/spatial
  support）、`panorama_station_closure.rs`（737，robust closure solve）、`panorama_station_closure_gate.rs`
  （637，闭环采样、修正上限、acceptance 与 relation 拒绝）、`panorama_station_closure_refine.rs`（749，闭环
  driver）、`panorama_station_topology.rs`（258，生成树、topology 观测、连通性）、`panorama_station_pose_seed.rs`
  （339，translation 种子、roll 校正、Local_Scale guard）、`panorama_station_solver.rs`（899，共享 solver）、
  `panorama_station_solve_entry.rs`（106，证据类型入口）和 `panorama_station_residual.rs`（177，Residual_Warp
  证据）。只移动代码，canonical relation gate、阈值、日志和报告字段不变；`panorama_stitching.rs` 由 36,356 行
  降到 32,184 行。警告集合与 `v1` 一致，release 全量库测试两次 `901 passed, 0 failed, 40 ignored`，station 相关
  定向测试、Node 契约和数据集契约（含输入 hash）通过；兰亭 10% 与马远 PSD 10% 回归的 preview 和六个裁片与
  `v1` 基线逐字节一致。同时修复了 `v1` 上约 50% 概率失败的 `world_resampled_reference_*` 测试隔离问题（残留的
  进程级 tone run report）。
- [x] N1-8 focus grouping 拆分：新增 `panorama_station_grouping.rs`（721 行，pairwise station evidence、
  局部 bracket 运动判定、verified scan overlap、按运动拆分组件和 grouping outcome）与
  `panorama_capture_groups.rs`（464 行，`FocusCaptureGroup`、binding evidence、bracket pose 锁定、capture group
  构建、anchor 规范化和渲染顺序）。只移动代码，移出的 1,089 行非空代码与新模块逐行一致；facade 降到
  31,077 行。警告集合与 `v1` 一致，release 全量库测试 `901 passed, 0 failed, 40 ignored`，grouping/
  station_evidence/capture_group/split/bracket/anchor 定向测试 `3/2/9/5/3/10 passed`，Node 契约通过；兰亭与
  马远 PSD 10% 回归的 preview 和六个裁片与 `v1` 基线逐字节一致。
- [x] N1-9 `panorama_utils/stitching.rs` 拆分：两个并行计划按不相交区间把 18,898 行的 `stitching.rs` 拆成 574 行
  facade 和 `panorama_utils/stitching/` 下的 34 个子模块：25 个生产模块（`canvas_maps` 345、`virtual_tiles` 350、
  `exposure` 431、`station_seam_sampling` 539、`station_seam_plan` 793、`station_seam_apply` 639、
  `progressive_seam` 680、`panorama_seam_band` 337、`layered_compositor` 441、`tile_ownership_stitcher` 1,363、
  `focus_layers` 496、`focus_color` 639、`focus_decision` 304、`focus_masks` 492、`focus_alignment` 909、
  `focus_foreground` 533、`focus_analysis` 429、`filters` 511、`canvas_fill` 441、`focus_tone` 717、
  `focus_seam_blend` 489、`focus_stack_entry` 415、`focus_stack_stitcher` 972、`seam_path` 388、`sampling` 161）
  和 9 个测试专用模块（`station_stroke_match`、`station_seam_diagnostics`、`test_tone_capture`、
  `final_world_owner_runner`、`score_noise_probes`、`score_noise_diagnostics`，以及外置的 `interpolation_tests`、
  `station_degradation_wiring_tests`、`focus_fusion_test_access`）。只移动代码：外部 `stitching::X` 路径通过重导出
  保持不变，函数体只做 43 处 `super::<兄弟模块>::` → `crate::panorama_utils::<兄弟模块>::` 路径限定。合并到
  `f5b4e555` 之上后警告集合与 `v1` 一致；`v1` 的 310 个非 `use` 条目逐项对应（305 个归一化后一致、3 个外置测试
  模块一致、2 个只差 rustfmt 闭包花括号）；release 全量库测试 `903 passed, 0 failed, 40 ignored`，14 个 Node 契约
  通过；兰亭与马远 PSD 10% 回归的 preview 和六个裁片与 `v1` 基线逐字节一致。仍超过 1000 行：
  `tile_ownership_stitcher.rs`（单函数 1,302 行，后续按阶段拆函数）、测试专用的 `final_world_owner_runner.rs`
  （1,067）和 `interpolation_tests.rs`（1,480，归 N2）。
- [x] N1-10 其余长文件拆分（2026-10-29）：`panorama_stitching.rs`（31,128 → 936 行 facade，38 个生产子模块 + 5 个外置
  测试模块）、`panorama_utils/mosaic.rs`（7,079 → 277）、`scan_mosaic.rs`（2,861 → 264）、`processing.rs`（1,299 → 98）、
  `stack_pipeline` 的十个生产文件（1,288–5,524 → 113–917，26 个生产子模块），以及应用层 export/exif/image_stack/
  style_transfer、image_processing/gpu_processing/lens_correction/lens_blur/denoising、file_management/mask_generation/
  ai_processing/crop_analysis/image_loader 与 `lib.rs`（子模块在 `src/lib/`，`generate_handler!` 中 52 个命令改为完整路径，
  命令名不变）。只移动代码：每个 track 的警告集合与基线一致、条目级核对 missing 0，release 全量库测试
  `903 passed, 0 failed, 40 ignored`，Node 契约通过，兰亭与马远 PSD 10% 回归 identical。约定、模块地图和仍超过 1000 行
  的文件见 `docs/source-layout.md`；按 2026-10-29 的决定，单个超长函数/组件不在本轮拆分。
- [x] `panorama_stitching.rs` 只保留稳定入口、阶段编排、常量和重导出。
- [x] `stitching.rs`、`mosaic.rs`、`processing.rs` 和 `stack_pipeline` 生产模块拆成单一职责 sibling
  模块；facade 不承载算法实现。`stitching.rs` 已在 N1-9 完成（facade 只保留四个共享像素 helper）；`mosaic.rs`、
  `processing.rs` 与 `stack_pipeline` 生产模块已在 N1-10 完成。
- [ ] 所有目标生产文件不超过 1000 行；N1-6 新增模块已经满足，剩余历史大文件和测试支持文件按 N1-7/N2
  继续拆分；模块 import 方向通过源码检查，禁止测试支持反向依赖生产。2026-10-29：除 `panorama_stitching/stitch_images.rs`
  与 `stitching/tile_ownership_stitcher.rs`（各只含一个超长函数，按决定保留）外，生产 Rust 文件均不超过 1000 行；
  import 方向的源码检查尚未建立。
- [x] 更新并维护 `docs/focus-stack-architecture.md`，记录当前 N1-6 模块树、阶段边界、验证命令和实际回归
  证据；不复制其他 worktree 的完成数字。
- [ ] 每完成一个职责层，运行该层定向测试并记录迁移前后输出等价性。

**N1 通过条件**：生产模块树可由架构文档生成；入口、缓存、solver、compositor 和 output 的边界清晰；
结构检查、编译和现有行为回归全部通过。

## N2：拆分测试支持和研究 oracle

- [x] 先拆 `stack_pipeline/test_support.rs` 和共享 fixture，再拆 focus ownership、registration/topology、
  alignment、Virtual_Tile、acceptance 和 interpolation 测试。2026-10-29：`test_support.rs`（2,150 → 78 行 facade + 5 个
  生成器模块）、`alignment_tests`、`acceptance_tests`、`panorama_virtual_tile_tests.rs`、`interpolation_tests`、
  `mosaic/tests`、`focus_fuser/tests`、`quality_gate_runner/tests`、`export_processing/tests` 与 `mask_generation/tests`
  均按主题拆分；测试名和总数（943）不变，路径对照见 `docs/test-path-map.tsv`。
- [x] `properties.rs` 按领域拆分；每个 property 保留原名称、过滤路径、生成器语义和失败样本。2026-10-29：
  13,683 行拆成 598 行 facade 与 22 个 Correctness Property 领域模块（最大 870 行），52 个 `proptest!` 块整块移动；
  测试名不变，路径保留 `stack_pipeline::properties::` 前缀，`package.json` 以 `--exact` 调用的两个 harness 留在 facade；
  `properties.txt` 的 8 条种子复制到每个含 proptest 的子模块对应路径。
- [ ] `panorama_structural_oracle.rs`、`owner_quality_oracle.rs`、`final_world_owner_oracle.rs` 拆成不超过
  1000 行的 test-only 模块，禁止从生产模块 import。2026-10-29：三者都已不超过 1000 行（structural 与 final-world 已
  拆分，owner_quality 为 889 行）；`stitch_images.rs` 与 `quality_gate_runner` 仍在 `#[cfg(test)]` 分支中调用
  structural oracle 和 acceptance helper，import 方向待处理。
- [ ] 将 `tests/focus-stack-quality-contract.mjs` 中的 production invariant 与 research/test-only oracle
  断言分开；release contract 不再依赖已停止的 owner、tone 或 structural research。
- [ ] `stack-pair-evidence:check` 只作为 research harness，不能成为 release gate，除非 N3 重新定义并验收它。
- [x] 每次迁移先跑定向测试，再跑完整 `cargo test --lib`；测试迁移提交不得同时改变生产逻辑。2026-10-29 的测试
  拆分提交只移动测试代码，生产拆分（含 `impl GpuProcessor` 分块）单独提交。

**N2 通过条件**：测试支持和 oracle 均不超过 1000 行，失败能定位到一个领域，测试数量和过滤路径可追踪，
生产构建不携带研究实现。

## N3：建立新的观测与真实素材验收入口

- [x] 固定 `tests/acceptance/focus-stack-datasets.json` 数据集 manifest、输入扩展名/排除规则和兰亭参考
  成片；使用 `RAW_EDITOR_FOCUS_STACK_DATA_ROOT=/Users/dp/Downloads/书画拼图 npm run
  focus-stack-datasets:check -- --require-data --hash-inputs` 校验数量、输入 digest 和参考文件。
- [ ] 在 manifest 基础上固定 Langyuan-10、Wenyuan-10 和 84 张 NEF 的解码状态、network=0、输出目录和
  运行参数；不把旧临时路径或旧报告 hash 写入任务状态。
- [x] PSD 导入按“一文件一张合成源图”完成解码、目录扫描和文件选择器接线；用马远 14 张 PSD 做批量
  回归，确认不展开图层且输出可进入现有拼接流程。当前回归为 14/14 decoded、13/13 连续邻接边，
  10% 输出 `1397×1173`，每个输出像素均有 source ownership；Quality Gate 因无参考 ROI 保持
  `insufficient_evidence`。
- [x] 真实素材回归入口：`npm run focus-stack-regression` 按 manifest 解析输入（含 `langyuan-10`、`wenyuan-10`
  冻结子集），运行 ignored harness，检查输入完整性、streaming/ownership 证据、参考文件 digest，并对照
  `tests/acceptance/focus-stack-regression-baselines.json`；test-only `panorama_acceptance_metrics.rs` 把输出配准到
  参考成片，按内容 tile 和左/中/右三段比较对比度归一化梯度能量。`v1` 的兰亭/马远 10% 基线已记录。
- [ ] 每个职责层完成后先运行上述数据契约检查；涉及成片行为时，必须用兰亭 45 张真实输入运行
  `real_focus_stack_fixture_from_env`，将输出和 `RAW_EDITOR_STACK_REPORT_DIR` 都写入 `/private/tmp`，并
  检查 45/45 输入、整图与左中右清晰度裁剪、Coverage/Ownership 证据和参考成片对照；禁止覆盖
  `/Users/dp/Downloads/书画拼图/兰亭/DSC08854_FocusStack.jpg`。当前恢复路径仍允许 preview-only，
  但最终发布验收必须另行确认 `output.written` 和实际输出文件。
- [ ] 统一 Stack_Report：input、proxy、grouping、station relation、topology、closure、fusion、owner、
  residual warp、tone、seam、composition、quality gate、output、resources、degradation 和 provenance。
- [ ] 所有诊断使用最终画布坐标，同时保存 source/native/world 坐标和转换矩阵；报告记录失败阶段而不是只写
  一个最终 verdict。
- [ ] 先跑 10 张样本确认报告链路和证据完整，再跑 84 张确认机位图、资源和停止原因；旧验收结果只作为待复核
  的风险提示，不直接标记通过或失败。

**N3 通过条件**：同一 manifest 重跑可复现报告 schema、输入 digest 和输出 digest；真实运行不会因缺字段、
坐标混用或旧 artifact 误读而改变结论。

## N4：先解决 station geometry，再进入画质

- [ ] 核对 source/native/world 坐标、crop、proxy scale、先验布局和机位成员；用 source member 标识，不用
  文件顺序或动态 station 编号作为关系证据。
- [ ] 让 feature/matching、Virtual_Tile、canonical relation gate、prior-free search seed、polish 和
  topology 使用同一候选生命周期：proposal → measurement → shared gate → accepted/rejected。
- [ ] 记录 `no overlap`、support、bidirectional、grid、scale、hull、orientation 和 closure 各阶段，禁止
  直接放宽既有门槛或强制 bridge。
- [ ] 机位图达到目标连通性后，再启用 station pose、intra-station registration 和 residual warp；证据不足
  时保持 identity。

**N4 通过条件**：每套真实素材的机位图可解释、闭环进入可测量状态、独立 holdout 与 production relation
  一致，并且通过的关系不会因线程数或候选遍历顺序变化。

## N5：处理 focus ownership 和融合证据

- [ ] 以当前运行的 registration、最终 world、Coverage 和 Source_RAW 采样生成 owner evidence；记录 winner、
  runner-up、run-length、重复测量方差、ROI 数和 provenance。
- [ ] owner coverage 不能以牺牲 effective pixel、gradient/noise、boundary、ROI 数或 Source_RAW fidelity
  换取；smoothness、hard terminal、ROI reservation 和 content takeover 研究不得自动进入生产。
- [ ] focus warp、Residual_Warp、mask、透明区域和单一 owner 写入保持可追溯；任何 fallback 都写入报告。
- [ ] 只在真实关系和闭环证据稳定后评估跨站融合，先保留原有安全回退。

**N5 通过条件**：owner 结论能回放到最终像素和 Source_RAW，holdout 不劣化，Coverage/Ownership/provenance
  一致，且不依赖 test-only oracle。

## N6：seam、tone、compositor 和大图输出

- [ ] 机位间 seam 只在双方 Coverage 有效且有足够重叠时求解；GraphCut、DP 或中线 fallback 记录 solver、
  代价、超时和机位对，不把无 Coverage 像素纳入接缝搜索。
- [ ] Tone_Harmonizer 使用可解释的 shared-world evidence；空间 tone、owner-fidelity 不可行方案保持研究
  隔离，不以模糊边界掩盖真实色差。
- [ ] TileSource/TileCompositor 按需读取全分辨率块，验证 tile 与 whole-frame 的像素、mask、ownership、
  alpha 和 provenance 等价；单块失败不得发布部分结果。
- [ ] BigTIFF/Pyramidal TIFF 写出支持 tile、overview、位深、ICC、alpha、取消、fsync 和原子 rename；输出
  metadata 和编码失败写入报告。

**N6 通过条件**：Langyuan/Wenyuan 的 seam、tone、边界和输出证据可定位；tile 顺序、缓存命中和线程数不改变
  结果 digest；失败不会残留半成品。

## N7：Quality Gate 与发布边界

- [ ] 先让 Quality Gate 完整记录所有判据和证据；几何断连、closure 不可靠、owner shortfall、tone degraded、
  boundary 超限或证据不足时明确拒绝发布。
- [ ] 只有 Langyuan-10、Wenyuan-10 和 84 张在连续可复现实跑中满足几何、融合、owner、tone、boundary、
  output 和资源条件后，才切换 blocking mode。
- [ ] 发布只接受通过 gate 的最终文件；报告、诊断和失败原因可以保留，但不得把 preview、`not_run` 或旧
  report 复制成最终产物。

**N7 通过条件**：Quality Gate 的 verdict、output.written、报告和文件系统状态一致，失败路径可重复，且没有
  通过修改阈值或绕过证据获得“通过”。

## N8：确定性、资源和可选 provider

- [ ] 对 1/2/8/16 线程、重复运行、取消、内存超限、tile/whole-frame 等价和 provider fallback 做回归。
- [ ] Rust provider 保持默认；可选 Python/OpenCV 只在固定 proxy fixture 上比较匹配率、重投影、seam、tone、
  owner、RSS、耗时和输出 digest，未证明收益不得切换默认 provider。
- [ ] 统一 Rayon/OpenCV 预算、缓存 key、显式拷贝计数、RSS 采样和临时文件生命周期。
- [ ] 输出、报告排序、浮点归约、RANSAC seed、tile 顺序和失败原因保持确定性。

**N8 通过条件**：资源边界和确定性报告可自动比较，provider 失败有安全回退，长任务不会产生不可控的重复
  大数组或残留临时文件。

## N9：最后再扩展算法

- [ ] 真实门禁稳定后，再评估 FlatArtwork 的受限 similarity、轻微曲率的分段 homography、受约束非刚性模型
  和严格裁切/透明/边缘填充策略。
- [ ] 每个新模型必须有独立 baseline、holdout、失败回退、资源预算和开关；不得直接覆盖稳定生产路径。
- [ ] 通用 `processing.rs` 在业务职责稳定后按采样、滤波、颜色和几何 primitive 拆分，并保持公共契约不变。

## 阶段完成检查单

每个 N0–N9 阶段完成时必须同时提供：

1. 代码差异和职责边界说明。
2. 定向测试、完整回归和源码契约结果。
3. 若涉及真实素材，提供 manifest、digest、报告、输出 digest、资源和失败阶段。
4. 更新架构文档和本文件当前状态；只更新当前基线，不追加逐轮历史流水账。

最终完成的定义是：生产文件和测试支持文件均不超过 1000 行，模块依赖方向稳定，真实素材验收可复现，
Quality Gate 与发布边界一致，且所有未验证研究代码都留在 test-only 或独立实验路径。
