# 书画多机位翻拍：产品分析、缺陷与开发任务

**书画堆栈的开发入口。** 先看 §1–§3 了解产品现状和问题，再按 §4 领任务；§6 起是规则和参考。其他类型的任务不在本文件：

| 类别 | 文件 |
| --- | --- |
| 总入口：整体状态、验收证据、文档导航 | [project-progress.md](../../../docs/project-progress.md) |
| 工程、代码结构、打包发布（ENG、原N0–N2） | [engineering.md](../../../docs/tasks/engineering.md) |
| 分区亮度与接缝诊断（BR） | [brightness-seam.md](../../../docs/tasks/brightness-seam.md) |
| 渲染、内存与色彩（RM、CL） | [render-color.md](../../../docs/tasks/render-color.md) |
| 堆栈旧条目（原PF、N3–N9）的完整说明，只读 | [stack-history.md](../../../docs/tasks/stack-history.md) |

技术合同在本目录 [requirements.md](requirements.md) 与 [design.md](design.md)。原《书画推进计划》已并入本文件并归档为 [artwork-stack-plan-2026-11-12](../../../docs/archive/artwork-stack-plan-2026-11-12.md)。

**怎么用**：一项任务交给一个 agent 一次完成，brief 按 §8 模板写。开始前检查：依赖已勾选；“冲突”里的任务没有在进行；需要的资源槽排得到（§6.1）。进行中的进展写在 lane 卡 `docs/artwork-lanes/<任务号>.md`；只有 L0（集成/守门）勾选本文件。`[x]` 表示已完成并合入 `v1`。

## 1. 产品分析

**定位（2026-11-11 用户确认）**：书画多机位翻拍专用工具。给定一幅书画的一次或多次翻拍（多机位、每机位多焦点层，可能混合焦段和相机），输出一张整幅清晰、无可见拼接块的原生分辨率成片。每个像素可追溯到源图；证据不足时明确说明哪里不足，不静默生成。固定机位微距等通用焦点堆栈只做回归保护，不追加能力。

**与开源和成熟软件比较**（详见[开源对比](../../../docs/focus-stack-open-source-comparison.md) §7）：开源的焦点堆栈工具（focus-stack、ChimpStackr、ShineStacker、PlaneFuse、OpenFocus）只处理固定机位；拼接工具（Hugin、OpenCV Stitcher、ASHLAR、BigStitcher）不处理每个机位内的焦点层。“多机位 × 多焦点层 × 平面书画 × RAW × 超大画布”这个组合目前没有现成项目覆盖。

| 方面 | 我们 | 开源最佳实践 | 判断 |
| --- | --- | --- | --- |
| 来源追溯与证据门禁 | 每个像素有 owner/Coverage/alpha 和 Stack_Report；无可验证重叠就拒绝；冻结二进制逐像素回归 | 弱 | 领先 |
| 书法笔画细节 | 硬 owner，不平均错位细节，避免双影 | 金字塔/加权融合在残差下出双影 | 领先 |
| 特征与配准召回 | FAST/BRIEF；阆苑 0–2 的 1,061 个探针只有 23 个通过 | SIFT/ECC/LightGlue/RoMa 可选 | 差距大 |
| 边接受阈值 | 按素材逐步校准的固定阈值 | 置换检验从数据估计（ASHLAR） | 差距中高 |
| 接缝与色调 | 低频 tone + 证据约束 owner 增益；曝光/反光差异下接缝可见 | 整图多频带 + 曝光/渐晕联合优化 | 差距中高 |
| 速度与资源 | 9 张 NEF 约 330–530 秒、9–12 GB；堆栈管线无 GPU | GPU/CPU 自动选择 | 差距大（任务不同，不能直接比） |
| 交互修复与复核 | 无修图入口，无对齐复核视图 | 来源锁定修图、多层输出 | 缺失 |
| 外部基准 | 无 | MFIF benchmark、交叉验证 | 缺失 |

**结论**：正确性和可追溯性是我们的长处，要保持；短板集中在配准召回、接缝色调、全量规模（内存和耗时）和面向用户的复核修图。改进顺序按“先让每套真实素材到 D5，再补产品能力”。

## 2. 当前缺陷

基于 `base-2026-11-12` 的实测（[base-p1 摘要](../../../docs/benchmarks/base-p1-codex-port.json)、[I1 摘要](../../../docs/benchmarks/i1-integration-baseline.json)）。

| # | 缺陷 | 证据 | 影响 | 对应任务 |
| --- | --- | --- | --- | --- |
| 1 | 兰亭纸面跨来源接缝变差 | 跨 owner 阶跃 P95=4（v63 为 3），最大 7；保护合同按用户决定暂放到 4 | 局部可见色阶 | 6、19 |
| 2 | 祁序 QG 不通过，导入对话框拒绝 | owner 清晰度覆盖 0.889、边界描线失败；scale 0.1 运行约 60 分钟、峰值 15.5 GB | 不能交付 | 11 |
| 3 | 枫鹰 QG 不通过 | 局部尺度比、边界描线失败，owner 证据不足，112 像素未解析；110 个输入中 6 张未进入排序 | 不能交付 | 12 |
| 4 | 全量素材从未在当前版本跑过 | 阆苑 84、文苑 126、祁序 1/2 单独、睡觉 45、马远2 都没有当前结果；内存是否超门未知 | 不知道哪些素材能用 | 5.1–5.6 |
| 5 | 阆苑几何余量极小、配准召回低 | 10 张输出 P95 距 3px 门槛约 0.012px；0–2 探针 23/1,061 通过 | 扩到 84 张可能不过 | 7、8、13 |
| 6 | 并发负载下结果可能不同 | I1f 的 9 张热跑与枫鹰并发时站点关系内点不同而失败，独立重跑通过 | 多 agent 并行时回归可能误报 | 4、22 |
| 7 | 规模化的速度与内存 | 10 张约 8 分钟；84/126 张没有数据；无按需 tile、无 GPU | 大素材可能跑不完 | 15 |
| 8 | 固定阈值迁移性差 | 3px、1%、相关系数 0.8/0.9 等按素材校准 | 新素材容易误拒或误收 | 7 |
| 9 | 缺少面向用户的复核与修图 | 没有对齐复核视图、失败说明和来源锁定修图 | 用户无法自己定位和修正问题 | 9、20、21 |
| 10 | 无参考图的素材只能靠人看 | 祁序、马远2、睡觉、文苑没有参考图 | D4 依赖用户时间 | §6.3 复看队列 |
| 11 | 诊断与失败状态不统一 | Stack_Report 诊断和最终发布状态未覆盖所有路径（原 N3.3、N7.3） | 失败难定位 | 储备 R1、R6 |

## 3. 改进计划

**路线**：基础设施（第 0 批）→ 摸清每套素材现状并修兰亭（第 1 批）→ 按素材逐套推进到 D5（第 2 批）→ 补产品能力（第 3 批）。每一批的目标：

| 批次 | 目标 | 退出条件 |
| --- | --- | --- |
| 第 0 批 | 多 agent 能安全地同时工作（一个编译、一个运行，排队，不占满机器） | 任务 1–4 完成，演练通过 |
| 第 1 批 | 每套素材有诊断卡；兰亭接缝回到 P95≤3；注册与阈值的离线证据 | 任务 5–9 完成 |
| 第 2 批 | 祁序、枫鹰、阆苑 84、文苑 126 等逐套到 D3/D4/D5 | 至少 3 套素材 D5 |
| 第 3 批 | 修图、复核视图、确定性矩阵、书画几何 | 视诊断与用户需求 |

**单套素材交付等级**：逐级推进，只有 D5 才算这套素材交付；任何回退都记录原因。

| 等级 | 名称             | 判据                                                                                      |
| ---- | ---------------- | ----------------------------------------------------------------------------------------- |
| D1   | 能出预览         | scale≤0.25 运行完成；每张源图要么参与，要么被明确排除并写明原因；失败时能定位到失败阶段   |
| D2   | 原生出图         | scale=1 写出 RGBA16 TIFF，来源 Coverage/owner/alpha 合同通过，`output.written` 与文件一致 |
| D3   | 门禁通过         | Quality Gate pass（足量 ROI），闭环 converged 或明确不适用；未通过项不靠改阈值消除        |
| D4   | 用户视觉确认     | 用户看过整图和关键原生裁片并确认；无参考图的素材也必须由用户确认                          |
| D5   | 可复现且资源合规 | 同一冻结二进制独立运行两次输出摘要一致；峰值 RSS 不超过资源门；回归保护已冻结             |

**资源门（本机 Apple M5，10 核，32 GiB）**：单次运行峰值 RSS ≤ 24 GiB（25,769,803,776 B），超过即判资源失败。耗时暂不设硬门槛，全量摸底后由用户给出范围。

**资源门（本机 Apple M5，10 核，32 GiB）**：单次运行峰值 RSS ≤ 24 GiB（25,769,803,776 B），超过即判资源失败。耗时暂不设硬门槛，全量摸底后由用户给出范围。

### 3.1 基础版本与素材

- **基础版本**：`v1` 上的标签 `base-2026-11-12`（I1 + codex 移植，回归输出与 I1 逐字节相同，见 [base-p1 摘要](../../../docs/benchmarks/base-p1-codex-port.json)）。之后每次 L0 合入并冻结的版本都直接落在 `v1` 上；其他历史分支不作开发起点。
- 素材目录 `/Users/dp/Downloads/书画拼图`（只读），清单 `tests/acceptance/focus-stack-datasets.json`。

| 数据集              | 输入         | 当前等级（base-2026-11-12） | 已知情况                                                                                         |
| ------------------- | ------------ | --------------------------- | ------------------------------------------------------------------------------------------------ |
| lanting 兰亭        | 45 JPEG      | D3                          | QG pass/8ROI；用户目视 I1 略好于 v63（未作 D4 确认）；纸面跨 owner 阶跃 P95=4，目标 ≤3；有参考图 |
| langyuan 阆苑女仙   | 84 NEF       | 前 10 张 D3                 | 10 张两次一致；输出 P95 距 3px 门槛约 0.012px；84 张未跑                                         |
| wenyuan 文苑图      | 126 NEF      | 前 10 张 D3                 | 126 张未跑，最大的一套                                                                           |
| qixu 祁序（1+2）    | 75 + 98 NEF  | 联合 0.1 能出图，QG fail    | owner 清晰度覆盖 0.889、边界描线失败；两次拍摄相隔两天；无参考图                                 |
| fengying 枫鹰雉鸡   | 63 + 17 + 24 | 0.1 能出图，QG fail         | 局部尺度比、边界描线失败，owner 证据不足，112 像素未解析；头部倍率 1.4 倍                        |
| mayuan-2 马远2      | 47 NEF       | 无                          | 未建基线                                                                                         |
| sleeping 睡觉       | 45 NEF       | 9 张子集 D5                 | 9 张是全局回归保护；45 张未跑                                                                    |
| mayuan-psd 马远 PSD | 14 PSD       | 暂缓                        | 2026-11-12 用户决定先不做；不进入任何任务和矩阵                                                  |

## 4. 任务清单

### 已完成

- [x] S0 / ENG.6 两条代码线集成，冻结 I1（2026-11-12）。见[I1 摘要](../../../docs/benchmarks/i1-integration-baseline.json)。
- [x] ENG.7 codex 分支评估与移植，打标签 `base-2026-11-12`（2026-11-12）。见[base-p1 摘要](../../../docs/benchmarks/base-p1-codex-port.json)。

### 第 0 批：并行基础设施（N8.5）

先完成本批，之后的任务才按 §6 的槽和目录规则运行。任务1的资源槽与冻结工具已验收，使用方式见 [stack-lab说明](../../../scripts/stack-lab/README.md)。任务2的单次运行、矩阵与内存门、任务3模板与复看队列、任务4线程预算修复与真实两槽演练均已验收。第0批任务1–4完成，后续按槽进入第1批。

- [x] 1. 资源槽与冻结二进制工具（2026-11-17）
  - 目标：`slot.py build|run|exclusive|status` 与 `freeze.py --label <label> [--worktree <path>] [--adopt <exe>]`，行为按 §6.1、§6.2。
  - 修改：`scripts/stack-lab/slot.py`、`freeze.py`、`common.py`。
  - 资源：`build` 槽（冻结一次）。依赖：无。冲突：无。
  - 完成标准：两个同类请求确实排队，`status` 显示占用者；上限与 `nice` 生效；用 `output/acceptance/base-p1/candidate-release-binary` 冻结出 `binaries/base-2026-11-12`，SHA 与 base-p1 一致；label 重复时拒绝。
  - 验收：17项隔离进程测试通过；真实base冻结SHA与原记录一致、原始提交及523项清单保留；真实Cargo编译/冻结、执行入口及3项定向冒烟通过。见 [任务卡](../../../docs/artwork-lanes/1-resource-freeze.md)及[机器摘要](../../../docs/benchmarks/task-1-resource-freeze.json)。此条仅验收任务1；后续状态见各条，不宣称已完成完整线程确定性演练。

- [x] 2. 单次运行与回归矩阵脚本（2026-11-17）
  - 目标：`run.py`（一次素材运行，自动排队 `run` 槽，写 `execution.json`/`analysis.json`：QG、未通过项、来源计数、峰值 RSS、内存门、wall、`contended`、TIFF SHA；兰亭取景运行附纸面审计和严格保护）和 `matrix.py --tier quick|medium|extended|full`。
  - 修改：`scripts/stack-lab/run.py`、`matrix.py`、`lanting_audit.py`，新数据文件 `tests/acceptance/regression-matrix.json`（每套素材一行预期：TIFF SHA、QG、闭环、已知失败的失败特征）。
  - 资源：`run` 槽。依赖：1。冲突：无。
  - 完成标准：对 `base-2026-11-12` 跑一遍中等层，全部与 [base-p1 摘要](../../../docs/benchmarks/base-p1-codex-port.json) 一致；故意给错一条预期时矩阵报失败并自动用 `exclusive` 重跑。
  - 验收：32项工具测试、快速层7项、中等层5/5通过；5份TIFF均与冻结基线相同，兰亭严格保护通过。独立副本故意写错SHA，两轮真实运行（run→exclusive）均保留该差异，矩阵按预期返回1。7次运行最大RSS10.371GiB≤24GiB，最终产物复核通过。见[任务卡](../../../docs/artwork-lanes/2-run-matrix.md)及[机器摘要](../../../docs/benchmarks/task-2-run-matrix.json)。此条仅验收任务2；任务3–4以各条状态为准。

- [x] 3. lane 卡与复看队列模板（2026-11-17）
  - 目标：`docs/artwork-lanes/README.md`（lane 卡模板、诊断卡模板、新素材接入步骤）和 `docs/artwork-lanes/review-queue.md`（待看 ≤3 套、按批记录）。
  - 修改：`docs/artwork-lanes/`。资源：无。依赖：无。冲突：无。
  - 完成标准：诊断卡字段覆盖 §4 任务 5 的全部输出项；复看队列能看出当前待看几套、每批包含哪些路径。
  - 验收：lane/诊断卡与新素材步骤齐全，覆盖任务5的9组输出字段；复看队列含当前表、批次和历史，跨批次待看≤3且绑定整图/原生裁片路径与版本。当前已登记0/3，不自动授予D4。17项文档核对通过；纯文档不重跑素材。见[模板说明](../../../docs/artwork-lanes/README.md)、[复看队列](../../../docs/artwork-lanes/review-queue.md)及[任务卡](../../../docs/artwork-lanes/3-templates.md)。该条仅验收任务3模板；任务4以其记录为准。

- [x] 4. 线程上限核对与两槽演练（2026-11-17）
  - 目标：确认 §6.1 的上限不改变基线，并实际演练一次“一个编译 + 一个运行 + 一个排队”。
  - 内容：9 张在 `RAYON_NUM_THREADS=5` 下运行，核对 TIFF 是否仍等于 v28；同时在 `build` 槽做一次冻结，再提交第二个运行请求验证排队；核对 `contended` 标记和输出隔离。
  - 资源：`build` + `run`。依赖：1、2。冲突：无。
  - 完成标准：结果写入 lane 卡；L0 勾选本批任务 1–4。若 TIFF 不同，记为 N8.2 缺陷，基线运行改用 `exclusive` 默认线程，并请用户决定。
  - 验收：修复独立预处理池6线程绕过Rayon=5的问题；旧冻结二进制无合同验证时强制exclusive。r2真实编译+运行+排队、nice、contended和输出隔离全部通过；两次9张的全局/预处理均5，TIFF=v28。快速层7项、Rust1095/0/44、35项工具测试、中等层5个范围均通过且TIFF保持冻结值，最大RSS11.876GiB。见[任务卡](../../../docs/artwork-lanes/4-two-slots.md)及[机器摘要](../../../docs/benchmarks/task-4-two-slots.json)。不外推N8.2完整确定性或其他全量素材。

### 第 1 批：第 0 批完成后

本批任务修改区域互不重叠，但都要排队用槽：任务 5 的各项和任务 6 的兰亭运行共用 `run` 槽，任务 6、7、8 与接缝诊断共用 `build` 槽。

- [ ] 5. 全量摸底（N3.5，只跑不改代码）
  - 每项一套素材，彼此独立，可由不同 agent 领取；都用 `binaries/base-2026-11-12`，scale 0.25，产出一张诊断卡（参与/排除的源图及原因、实际路径和 station 数、停止阶段、N4.5 失败链计数、峰值 RSS、wall、预览和最明显问题的位置、初步判断归哪个区域组、是否需要任务 7–8 或 15 的横向能力）。
  - 修改：只写 `docs/artwork-lanes/S2-<dataset>.md` 和 `docs/benchmarks/`。冲突：无（只排队）。
  - [ ] 5.1 马远2 47 张（`run`）
  - [ ] 5.2 睡觉 45 张（`run`）
  - [ ] 5.3 祁序1 75 张、祁序2 98 张分别跑（祁序2 首次 `exclusive`）
  - [ ] 5.4 枫鹰三文件夹（`run`，与 base 的 0.1 结果对照）
  - [ ] 5.5 阆苑 84 张（`exclusive`）
  - [ ] 5.6 文苑 126 张（`exclusive`）

- [ ] 6. 兰亭纸面接缝（N6.5，流式 mosaic 组）
  - 目标：纸面跨 owner 阶跃 P95 从 4 降回 ≤3，边界 >6 处数不增加，QG 仍 pass，严格保护通过；视觉变化进入复看队列。
  - 修改：`mosaic/streaming_mosaic.rs`、`streaming_illumination.rs`、`quality_gate_runner/*`。
  - 资源：`build` + `run`。依赖：1、2。冲突：任务 11、12（同区域）。

- [ ] 7. 置换检验离线实验（PF.8）
  - 目标：用已保存的匹配点和代理图，统计随机不相邻机位对能达到的相似度分布，与现有固定门槛对照，看能否识别重复题字/人物造成的假边；阆苑 0–2 为第一条测试边。只写 test-only 代码和报告，不改生产路径。
  - 修改：注册实验区域（新测试模块）、`docs/`。资源：少量 `build`。依赖：1。冲突：无。

- [ ] 8. 注册 provider 离线对照（PF.10 注册部分）
  - 目标：FAST/BRIEF、SIFT 或 AKAZE、ECC 精修、LightGlue+DISK/ALIKED（ONNX）对照，指标见[开源对比 §10.2](../../../docs/focus-stack-open-source-comparison.md)；provider 只提交候选，最终门禁不变；不用 SuperPoint 权重和 MASt3R。
  - 修改：注册实验区域中与任务 7 不同的文件、`docs/`。资源：少量 `build`。依赖：1。冲突：无。

- [ ] 9. 来源锁定修图设计（N5.5，只写文档）
  - 目标：交互与数据合同：owner 覆盖层格式、可选来源的判定（有效 Coverage 且已配准）、撤销、导出与 Stack_Report 记录、人工区域不计入自动 QG。
  - 修改：`docs/` 新文件。资源：无。依赖：无。冲突：无。

- 10. 书画接缝诊断（可选）：在 [brightness-seam.md](../../../docs/tasks/brightness-seam.md) 的 BR.5–BR.7 跟踪，本文件不另勾选。修改前端与诊断区域，少量 `build`，与本批其他任务无冲突。

### 第 2 批：按诊断卡修复（需要对应的任务 5 诊断卡）

每项的具体目标在诊断卡出来后由 L0 写入 brief。同一区域组内一次只有一项在改代码，其余可以先做诊断。

- [ ] 11. 祁序到 D3（流式 mosaic 组）
  - 依赖：5.3、6 已合入。冲突：任务 6、12。
- [ ] 12. 枫鹰到 D3（流式 mosaic 组）
  - 依赖：5.4、11 已合入。冲突：任务 6、11。
- [ ] 13. 阆苑 84 张到 D2/D3（站点求解组）
  - 依赖：5.5。冲突：任务 14、15。
- [ ] 14. 文苑 126 张到 D2/D3（站点求解组）
  - 依赖：5.6、13 已合入。冲突：任务 13、15。
- [ ] 15. 必要性能（PF.9，含未移植的 5 个接缝提速提交）
  - 只在诊断卡显示超内存门或无法跑完时启动；接缝提速修改 `stitching/station_seam_*`，属站点求解组。计时只在 `exclusive` 下。依赖：相应诊断卡。冲突：任务 13、14。
- [ ] 16. 马远2、睡觉 45 张推进
  - 依赖：5.1、5.2。按诊断卡归入流式 mosaic 组或站点求解组的队列，冲突随所归的组。
- [ ] 17. 新素材接入（以后每套新素材一项）
  - 登记 manifest → D1 → 诊断卡 → 归组。修改：`tests/acceptance/focus-stack-datasets.json`、`docs/artwork-lanes/`。冲突：无。

### 第 3 批：至少 3 套素材达到 D5 之后

- [ ] 18. 书画几何（N9.1），仅当诊断卡显示纸面起伏导致失败
- [ ] 19. OwnerHybrid：笔画/印章硬 owner，纸面多频带
- [ ] 20. 来源锁定修图实现（N5.5，按任务 9 的设计）
- [ ] 21. 对齐复核视图与失败说明 UI
- [ ] 22. 确定性矩阵（N8.2）：线程数、改名乱序、取消、内存超限、并发负载
- 桌面完整流程与发布在 [engineering.md](../../../docs/tasks/engineering.md) 的 ENG.3、ENG.4 跟踪。

## 5. 储备任务（原 PF、N3–N9 中尚未排进 §4 的未完成项）

这些条目的完整说明和历次记录在 [stack-history.md](../../../docs/tasks/stack-history.md)。排进 §4 时由 L0 写成正式任务并在这里勾选；已经由 §4 任务覆盖的原条目不再列出（PF.8 置换检验→7，PF.10 注册→8、OwnerHybrid→19，PF.9→15，N3.5→5，N5.5→9/20，N6.5→6，N7.5→§3 交付等级，N8.2→22，N8.5→1–4，N9.1→18）。

- [ ] R1. 全路径统一 Stack_Report 与最终画布诊断（原 N3.3，[说明](../../../docs/tasks/stack-history.md#n3)）
- [ ] R2. 每次行为变化先定数据契约再做受影响真实回归（原 N3.4，[说明](../../../docs/tasks/stack-history.md#n3)）
- [ ] R3. 跨素材统一 proposal→measurement→shared gate 及完整失败链（原 N4.5，[说明](../../../docs/tasks/stack-history.md#n4)）
- [ ] R4. 每套素材通过关系与几何门禁后才用对应 pose/residual，否则安全回退（原 N4.6，[说明](../../../docs/tasks/stack-history.md#n4)）
- [ ] R5. owner 证据、回放与 fallback 的跨素材核对（原 N5.2、N5.3，[说明](../../../docs/tasks/stack-history.md#n5)）
- [ ] R6. 所有失败准确反映在最终发布状态，preview/not_run 不冒充通过（原 N7.3，[说明](../../../docs/tasks/stack-history.md#n7)）
- [ ] R7. 满足连续可复现门槛后再评估全面 blocking mode（原 N7.4，[说明](../../../docs/tasks/stack-history.md#n7)）
- [ ] R8. 跨机位 seam 只在双方有效 Coverage 内求解并记录 fallback（原 N6.2，[说明](../../../docs/tasks/stack-history.md#n6)）
- [ ] R9. Tone_Harmonizer 的 shared-world 证据与 owner fidelity 泛化验证（原 N6.3，[说明](../../../docs/tasks/stack-history.md#n6)）
- [ ] R10. 按需全分辨率 tile 与 BigTIFF/Pyramid 输出合同（原 N6.4；超内存时并入任务 15，[说明](../../../docs/tasks/stack-history.md#n6)）
- [ ] R11. 跨站融合评估（原 N5.4，几何闭环稳定后，[说明](../../../docs/tasks/stack-history.md#n5)）
- [ ] R12. 代理图与 feature 跨运行缓存（原 PF.7，[说明](../../../docs/tasks/stack-history.md)）
- [ ] R13. coarse retrieval→top-K 候选与 bounded rescue（原 PF.8 检索部分，[说明](../../../docs/tasks/stack-history.md)）
- [ ] R14. 统一 Rayon/OpenCV 预算、缓存 key、RSS 采样与确定性检查（原 N8.3，[说明](../../../docs/tasks/stack-history.md#n8)）
- [ ] R15. Python/OpenCV 固定 fixture 对照，未证明收益不切默认（原 N8.4，[说明](../../../docs/tasks/stack-history.md#n8)）
- [ ] R16. 新模型的独立 baseline、holdout、回退和开关规范（原 N9.2，[说明](../../../docs/tasks/stack-history.md#n9)）
- [ ] R17. scene-linear、GPU 与 learned fusion 研究（原 PF.11，暂缓，[说明](../../../docs/tasks/stack-history.md)）

## 6. 并行规则

### 6.1 资源槽（2026-11-12 用户确认）

- 复杂任务排队；同一时刻最多“一个开发、一个测试”，任何时候都不占满整台机器。
- **`build` 槽**（编译、Rust 单元测试）和 **`run` 槽**（真实素材运行、回归矩阵）各 1 个。同时需要两者的任务用 **`exclusive`**：84/98/126 张首次运行、性能计时、预估峰值 > 20 GiB 或没有历史峰值的运行。
- 读代码、写设计、整理报告、看已有输出不占槽，但不得编译或跑素材。
- 上限：`build` ≤ 3 个编译作业，`run` ≤ 5 个渲染线程，都以 `nice` 运行，至少留 2 核；`exclusive` ≤ 8 线程 / 6 作业。上限是否改变基线输出由任务 4 核对。
- 内存：`run` 槽单次预估峰值 ≤ 20 GiB。已知峰值：9 张 NEF 8.5–12.5 GB，阆苑/文苑 10 张 9–10 GB，兰亭 scale=1 9.7–11.4 GB，祁序/枫鹰 scale=0.1 14.8–15.5 GB。
- 拿不到槽就阻塞等待并显示占用者；槽状态和历史在 `output/acceptance/.slots/`。
- `run` 运行期间若 `build` 也被占用，报告标 `contended: true`，耗时只用于判断能否跑完。性能对比只在 `exclusive` 下做，沿用 PF.6 交错轮次。
- 回归矩阵结果与预期不符时，先用 `exclusive` 重跑一次再判定（I1f 曾在并发负载下出现站点关系不一致）。

### 6.2 代码、分支与编译

- 改代码的任务在 `.worktrees/<任务号>` 开 worktree，分支 `lane/<任务号>/<topic>`，从 `v1` 最新冻结提交开出。
- **所有 worktree 共用编译目录** `CARGO_TARGET_DIR=/Users/dp/Desktop/raw-editor/src-tauri/target`，只在 `build` 槽内编译。编完立即把测试可执行文件冻结到 `output/acceptance/binaries/<label>/`（含提交、SHA、源码清单）；运行只用冻结副本。label 不可复用。
- **代码区域组**：修改同一区域的任务先后进行，不同区域可以同时进行。

| 区域组      | 主要文件                                                                                                             | 归属的素材任务   |
| ----------- | -------------------------------------------------------------------------------------------------------------------- | ---------------- |
| 流式 mosaic | `panorama_utils/mosaic/streaming_*.rs`、`frame_exposure.rs`、focus-scan 修补、`quality_gate_runner/*`                | 任务 6、11、12（兰亭、祁序、枫鹰） |
| 站点求解    | `panorama_station_*.rs`、`mosaic/native_control_points.rs`、`stack_pipeline/joint_warp*`、`stitching/station_seam_*` | 任务 13、14、15（阆苑、文苑、性能）       |
| 注册实验    | `processing/features.rs` 旁的 test-only 新模块                                                                       | 任务 7、8       |
| 前端与诊断  | `src/`、接缝诊断模块                                                                                                 | 任务 10          |
| 工具与文档  | `scripts/stack-lab/`、`tests/acceptance/*.json`、`docs/`                                                             | 任务 1–5、9     |

- 新素材接入后按诊断卡里的实际路径归入已有区域组，作为组内队列的一项，不新增 lane。
- `stitch_images.rs`、`stack_pipeline/report_schema_*.rs` 是全局热点：修改前在 lane 卡登记函数和原因，L0 发现重叠时安排先后。
- 任务内禁止修改共享质量阈值、全局默认开关和其他素材的基线；确需修改时写在 lane 卡上，由 L0 汇总请用户决定。

### 6.3 合入、输出与复看

- 只有 L0 向 `v1` 合入，每次一个任务：rebase 到 `v1` → 跑回归矩阵中等层及受影响素材 → 合入 → 冻结新版本并更新矩阵预期。摘要变化若是预期效果，需逐项说明并附视觉复看；否则拒绝。
- 运行输出放在 `output/acceptance/<任务号>/<dataset>-<版本>/`，任务之间不共用目录；机器摘要写入 `docs/benchmarks/<dataset>-<版本>-<主题>.json`。版本号按任务递增（如 `qixu-r1`），不与 v28–v63 混用。
- 磁盘：共用编译目录约 30 GiB 由 L0 管理；每个任务的输出和冻结二进制合计 ≤ 20 GiB；全局剩余低于 40 GiB 时暂停新运行，请用户决定清理范围。原生 TIFF 只保留每个等级的通过版本和最近一次失败版本。
- **复看**：同时等用户看的素材不超过 3 套，L0 按批汇总到 `docs/artwork-lanes/review-queue.md`（整图 + 关键原生裁片路径）；满 3 套时其余素材停在 D3，任务转去诊断下一套。

## 7. 回归矩阵与风险

L0 合入前必跑；任务提交合入请求前至少跑快速层和受影响素材。预期值放在数据文件里，新素材到 D5 后只加入全量层，中等层保持有界。

| 层级 | 内容                                                                                                                                                      | 估计耗时   |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| 快速 | `cargo fmt --check`、`cargo check`、Rust 库测试（当前 1095 项）、`focus-stack-quality:check`、`datasets:check`、`regression:check`、`lanting:guard:check` | 十几分钟   |
| 中等 | 9 张冷/热（TIFF=v28）；阆苑 10、文苑 10（TIFF=当前冻结摘要）；兰亭 45 取景 + 严格保护                                                                     | 约 45 分钟 |
| 扩展 | 祁序联合、枫鹰 scale 0.1（已知失败，核对失败特征不变或变好）                                                                                              | 约 80 分钟 |
| 全量 | 已达 D3 及以上的全量素材                                                                                                                                  | 随素材增加 |

### 风险与应对

| 风险                                      | 影响                   | 应对                                                                     |
| ----------------------------------------- | ---------------------- | ------------------------------------------------------------------------ |
| 84/126 张超过 32 GiB 内存                 | 无法到 D2              | 任务 5.5/5.6 先用 0.25 拿峰值；超限则任务 15 提前，优先按需 tile（N6.4） |
| 阆苑输出 P95 余量只有约 0.012px           | 扩到 84 张后可能不过   | 任务 13 先看全量里最差的边，不调门槛                                     |
| 无参考图的素材（祁序、马远2、睡觉、文苑） | D4 全靠用户看          | 复看队列按批确认                                                         |
| 多个任务同时改流式 mosaic                 | 合入冲突、回归无法归因 | 区域组内先后进行，一次只合入一项                                         |
| 并发运行导致计时或结果失真                | 结论不可信             | `contended` 标记；性能只在 `exclusive` 测；矩阵不符先 `exclusive` 重跑   |
| 用户复看跟不上                            | 结果堆在 D3            | 待看 ≤3 套，满了转去诊断                                                 |

## 8. 任务 brief 模板

```text
任务: <编号 名称>
目标: <把 <dataset> 从 D<n> 推进到 D<n+1>，或具体交付>
基础: v1 提交 <sha>，冻结二进制 output/acceptance/binaries/<label>/ (SHA-256 <...>)
工作区: /Users/dp/Desktop/raw-editor/.worktrees/<任务号>，分支 lane/<任务号>/<topic>
编译: CARGO_TARGET_DIR=/Users/dp/Desktop/raw-editor/src-tauri/target，只在 build 槽内
素材: /Users/dp/Downloads/书画拼图/<目录>（只读）
输出: /Users/dp/Desktop/raw-editor/output/acceptance/<任务号>/<dataset>-<版本>/
资源槽: <build / run（预估峰值 GiB）/ exclusive>
修改区域: <文件>；冲突任务: <编号>
已知情况: <诊断卡要点、相关 benchmark 路径>
禁止: 修改共享阈值/默认开关/其他素材基线；写入素材目录；直接合入 v1；勾选本文件或修改 docs/project-progress.md
完成时提交: lane 卡、docs/benchmarks/<...>.json、回归矩阵结果、需用户确认的问题
```
