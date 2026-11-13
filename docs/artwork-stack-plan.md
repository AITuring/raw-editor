# 书画多机位翻拍：推进计划与多 agent 并行方案

> [返回项目进度与任务总览](project-progress.md#next) · 专项文档。本文规定阶段、泳道、并行规则和交接格式；任务勾选状态和推进顺序仍只在[总览](project-progress.md)维护，本文不另设任务表。

制定日期：2026-11-11。依据：用户确认的产品定位（书画多机位翻拍专用工具），以及仓库当前状态。本文没有重新运行算法或验收；凡标为“估计”的数字都来自既有运行记录，阶段 2 实测后需要更新。

## 1. 目标与“完成”的定义

### 1.1 产品目标

给定一幅书画的一次或多次翻拍（多机位、每机位多焦点层、可能混合焦段和相机），输出一张整幅清晰、无可见拼接块的原生分辨率成片。每个像素可追溯到源图；证据不足时明确说明哪里不足，不静默生成。

固定机位微距等通用焦点堆栈只做回归保护，不追加能力。

### 1.2 单套素材的交付等级（新 N7.5）

每套素材按以下等级逐级推进，只有达到 **D5** 才算“这套素材交付”。等级只升不跳；任何一级回退都要在 lane 卡里记录原因。

| 等级 | 名称 | 判据 |
| --- | --- | --- |
| D1 | 能出预览 | scale≤0.25 运行完成，所有源图要么参与、要么被明确排除并写明原因；失败时有可定位的失败阶段 |
| D2 | 原生出图 | scale=1 写出 RGBA16 TIFF，来源 Coverage/owner/alpha 合同通过，`output.written` 与文件一致 |
| D3 | 门禁通过 | Quality Gate pass（足量 ROI），闭环 converged 或明确不适用，未通过项不靠改阈值消除 |
| D4 | 用户视觉确认 | 用户看过整图和关键原生裁片并确认；无参考图的素材也必须由用户确认，不能以指标代替 |
| D5 | 可复现且资源合规 | 同一冻结二进制独立运行两次输出摘要一致；峰值 RSS 不超过资源门（§1.3）；相应回归保护已冻结 |

### 1.3 资源门（本机）

本机为 Apple M5（4 性能核 + 6 能效核）、32 GiB 内存，2026-11-11 时数据盘剩余约 187 GiB。

- **内存门**：单次运行峰值 RSS ≤ 24 GiB，沿用兰亭既有的 25,769,803,776 B 门槛，为系统留余量。超过即判资源失败，不靠关闭其他程序过关。
- **耗时门**：暂不设硬门槛。阶段 2 拿到全量实测后，由用户给出可接受的范围再写入本节。目前已知 9 张 NEF 约 330–535 秒，10 张约 500–635 秒，兰亭 45 张 JPEG 约 336–466 秒。

## 2. 现状盘点

### 2.1 代码线（阻塞并行的首要问题）
**2026-11-12已解决：** 两条代码线已在`integrate/i1`合并、验证并快进合入`v1`，I1 = `b68c9e22`（集成二进制来自`c8fd7e6a`，SHA-256 `a282336d…`）。2026-11-12又从`codex/layered-focus-modularize-2026-10-26`移植3项修复，形成基础版本`base-2026-11-12`，回归输出与I1逐字节相同，此后各lane从该标签开始（见总览ENG.7）。下表为S0开始前的盘点，保留作记录。`fix/stack-excluded-owner-report-nan`与`codex/layered-focus-modularize-2026-10-26`两个分支保留，worktree已删除。

| 位置 | 内容 | 状态 |
| --- | --- | --- |
| `v1` 已提交（`e6ffdf7a`） | 公共基础 | — |
| `v1` 主工作区未提交修改 | 118 个文件，+8,191/−1,564 行：阆苑/文苑 v36–v47 几何与闭环、兰亭 v45–v60、QG runner、报告 schema | **未提交**，是阆苑 10/文苑 10/9 张/兰亭 v60 验收所用的代码 |
| `feat/stack-baselines`（`.worktrees/stack-gaps`） | 10 个提交（2026-10-31 至 11-09）：祁序 focus-scan 清晰度竞争、软区修补、多次拍摄融合、枫鹰三文件夹拼接与整组曝光、GUI 与 harness 一致、祁序/枫鹰回归基线 `tests/acceptance/focus-stack-regression-baselines.json`；worktree 另有 10 个未提交文件 | **未合入 v1** |
| 两者重叠 | 20 个文件同时改动，含 `streaming_mosaic.rs`、`streaming_store.rs`、`stitch_images.rs`、`source_prep.rs`、`image_stack.rs`、`degradation.rs`、`focus_stack_fixture.rs`、数据 manifest 和回归脚本 | 合并必然有实质冲突 |
| `.worktrees/fix-stack-excluded-owner`（`cd5c7db0`） | NaN station-relation 读回修复 | v1 已有同内容提交 `2f942a85`，疑为重复，删除前需确认 |
| `.worktrees/layered-focus-cleanup`（`12512982`） | 2026-10-26 的模块化清理分支 | 未合入，现状待确认 |

另外，`docs/project-progress.md`、`docs/focus-stack-open-source-comparison.md` 等多份文档没有纳入 git。

### 2.2 素材矩阵

源目录：`/Users/dp/Downloads/书画拼图`（manifest：`tests/acceptance/focus-stack-datasets.json`）。

| 数据集 | 输入 | 体积 | 当前最高等级 | 已知情况 |
| --- | --- | --- | --- | --- |
| lanting 兰亭 | 45 JPEG | 0.55 GB | D2（严格 QG 未过） | v60 主体色块回归通过；legacy 质量平面未接 QG；纸面略偏暖；有用户确认的参考图 |
| langyuan 阆苑女仙 | 84 NEF | 5.0 GB | 前 10 张 D3 + 重复一致 | 输出 P95 距 3px 门槛约 0.012px；0–2 窄重叠且跨倍率；84 张未跑 |
| wenyuan 文苑图 | 126 NEF | 7.0 GB | 前 10 张 D3 | 126 张未跑，是最大的一套 |
| qixu-1 / qixu-2 祁序 | 75 / 98 NEF | 4.5 / 5.7 GB | 分支上有预览基线 | 同一幅画、相隔两天的两次拍摄；分支有软区修补和多次拍摄融合；无参考图 |
| fengying 枫鹰雉鸡 | 63 + 17 + 24（ARW，底部混 NEF） | — | 分支上有 scale 0.1 预览基线 | 头部倍率是主体的 1.4 倍；110 输入中 104 进入排序；QG `insufficient_evidence`；走 `legacy_single_layer_mosaic`，74 个 station |
| mayuan-2 马远2 | 47 NEF | 2.8 GB | 无 | 未建基线 |
| mayuan-psd 马远 | 14 PSD | 19.8 GB | 10% 回归 | 单个 PSD 很大，解码内存风险高；QG 参考证据不足 |
| sleeping 睡觉 | 45 NEF | 2.6 GB | 9 张子集 D5 | 45 张全量未跑；9 张是现有最主要的回归保护 |

## 3. 阶段

| 阶段 | 内容 | 并行度 | 进入条件 | 退出条件 |
| --- | --- | --- | --- | --- |
| **S0 代码线集成** | 两条代码线合成一个可信基线 I1 | 串行，单 agent | 用户授权提交主工作区修改 | §4 全部完成，I1 冻结 |
| **S1 并行基础设施** | 运行槽、输出隔离、lane 卡、回归矩阵脚本 | 单 agent，可与 S0 后半程重叠 | S0.1 完成 | §5 规则可执行并经一次演练 |
| **S2 全量摸底** | 每套素材用 I1 跑 D1，定位失败阶段 | 多 lane，只跑不改代码 | S0、S1 完成 | 每套素材有一张诊断卡 |
| **S3 按素材修复** | 各 lane 诊断并提出修复，L0 逐个合入 | 诊断并行，合入串行 | S2 诊断卡 | 各素材逐级推进到 D3/D4 |
| **S4 书画专用能力** | 曲率模型、注册改进、置换检验、OwnerHybrid、接缝诊断 | 研究并行，合入串行 | S2 证明对应问题存在 | 有可复现收益的才进默认路径 |
| **S5 产品化** | 修图 N5.5、对齐复核与失败说明 UI、确定性矩阵、桌面与发布 | 部分并行 | 至少 3 套素材 D5 | 见 ENG.3/ENG.4 |

兰亭 N6.5 在 S0 之后马上作为 L1 继续，不等 S2。

## 4. S0 代码线集成（ENG.6）
**状态：2026-11-12完成。** 按用户选项A保留两条线行为；新增导入前拍摄参数扫描（参数不一致用逐帧曝光平面，一致则统一调整）；兰亭TIFF相对v63变化并待用户目视复核；祁序/枫鹰首次测得QG fail，用户选择严格合入，交L4/L5。结果与集成中修复见[I1机器摘要](benchmarks/i1-integration-baseline.json)和总览ENG.6。以下为原执行步骤。

一个 agent 串行执行，其他 lane 在 S0 完成前不改代码。

1. **冻结当前状态，不丢东西**
   - 为主工作区未提交修改建分支 `wip/v1-2026-11-11`，按主题分提交：几何闭环（N4.x）、兰亭 tone/owner（N6.5）、QG 与报告、测试支持、文档。**需要用户授权才能提交**；提交前把 `git status` 和分组方案给用户看。
   - 把 `docs/` 下未跟踪的文档一并纳入提交，至少包括 `project-progress.md`、两份开源对比、本计划和 `benchmarks/`。大文件和本地产物按 `.gitignore` 处理，不提交素材和 TIFF。
   - `.worktrees/stack-gaps` 的 10 个未提交文件同样先提交到 `feat/stack-baselines` 上，或明确丢弃并经用户确认。
2. **合并**
   - 以 `wip/v1-2026-11-11` 为基础合入 `feat/stack-baselines`，在新分支 `integrate/i1` 上进行。
   - 20 个重叠文件逐个解决冲突。原则：两边的行为都要保留；冲突处无法同时保留时，停下来把具体冲突和取舍方案交给用户，不自行选边。
   - 合并后检查长文件规则（N1.11）是否被破坏。
3. **验证（I1 回归矩阵，见 §6）**
   - 快速层全部通过。
   - 9 张冷/热 TIFF 与 v28 一致；阆苑 10、文苑 10 的 TIFF 与 v44/v47 摘要一致；兰亭 guard `--regression-only` 通过，TIFF 与 v60 一致；祁序/枫鹰分支基线的预览和裁片摘要一致。
   - 任何摘要变化都要定位到具体提交并解释，不能直接更新基线。
4. **冻结 I1**
   - 记录 I1 提交、`cargo test --release --no-run` 产物路径和 SHA-256，写入 `docs/benchmarks/i1-integration-baseline.json`。
   - 更新数据 manifest 和总览 §7.2：祁序/枫鹰的基线状态以合并后的事实为准。
5. **清理**：列出 `fix-stack-excluded-owner`、`layered-focus-cleanup` 两个 worktree 的处理建议，由用户决定是否删除。

风险：冲突集中在流式 mosaic 和 `stitch_images.rs`，这正是兰亭 v54–v60 和祁序/枫鹰都改过的地方。合并后兰亭 TIFF 不一致的概率不低；如果出现，要先判明是哪一边的行为变化，再请用户决定保留哪种。

## 5. 并行规则（N8.5）

### 5.1 泳道（lane）

| Lane | 范围 | 改代码 | 首要任务 | 主要触及文件（冲突热点） |
| --- | --- | --- | --- | --- |
| **L0 集成/守门** | 唯一能向 `v1` 合入的 lane；维护冻结二进制、回归矩阵、总览文档 | 只做合并和冲突修复 | S0；之后逐个合入其他 lane | 全部 |
| **L1 兰亭** | N6.5 | 是 | legacy 原生 owner/source/置信度质量平面接入 QG，使严格 guard 能判定 | `streaming_mosaic.rs`、`streaming_illumination.rs`、`quality_gate_runner/*` |
| **L2 阆苑** | 10 → 84 | 是 | 84 张 D1；用 N4.5 失败链定位不连通、窄重叠和跨倍率问题 | `native_control_points.rs`、`panorama_station_solver.rs`、`joint_warp*` |
| **L3 文苑** | 10 → 126 | 是 | 126 张 D1；首次运行独占内存槽，先测峰值 | 同 L2 |
| **L4 祁序** | 祁序 1、祁序 2，以及两次拍摄联合 | 是 | 合并后复验分支基线；原生 D2；对照 `.agents/artifacts/qixu-fusion-2026-11-08` 里的残余模糊区 | `streaming_mosaic.rs`、focus-scan 修补相关模块 |
| **L5 枫鹰** | 主体、头部、底部三文件夹 | 是 | 解释 110 输入中 6 张未进入排序的原因；QG 从 `insufficient_evidence` 推进到可判定 | 多次拍摄融合、`frame_exposure.rs` |
| **L6 马远** | 马远2（47 NEF）；马远 PSD 后置 | 是 | 马远2 建基线并到 D1 | 视诊断而定 |
| **L7 睡觉** | 45 张全量 | 是 | 45 张 D1；9 张子集继续作为全局回归保护，不在本 lane 修改其基线 | 视诊断而定 |
| **L8 注册与阈值** | PF.8 置换检验、PF.10 注册 provider 对照 | 先只做离线实验，合入前转为正式改动 | 用各 lane 保存的 proxy 和原生点做离线对照，阆苑 0–2 作为首条测试边 | `processing/features.rs`、`panorama_feature_pairs.rs` |
| **L9 性能** | 只做跑全量所必需的优化 | 是 | 等 S2 的 RSS 和耗时数据；所有计时在独占槽内完成 | 视 profile 而定 |
| **L10 接缝诊断** | BR.5–BR.7 | 是 | 书画真实接痕标注与负例集 | `src/` 前端与诊断模块，与堆栈管线冲突少 |
| **L11 书画几何** | N9.1 分段单应 / 受约束非刚性 | 先 test-only 研究 | 等 S2 指出哪些素材因纸面起伏失败 | 新模块为主 |

同一套素材同一时间只归一个 lane。L4（祁序）和 L5（枫鹰）都依赖分支上的 focus-scan 代码，两者改到同一模块时要先在 lane 卡上登记。

### 5.2 运行槽（内存与 CPU）

- 所有真实素材运行必须先领运行槽，跑完释放。锁文件放在 `output/acceptance/.run-slots/`，记录 lane、数据集、scale、PID、开始时间和预估峰值 RSS。
- **同时运行的预估峰值之和 ≤ 26 GiB。** 预估值取该数据集同 scale 的历史峰值；没有历史数据时按 24 GiB 计，也就是独占。
- 已知峰值：9 张 NEF 约 8.5–12.5 GB，阆苑 10 张约 9.4–9.7 GB，文苑 10 张约 9.1 GB，兰亭 scale=1 约 9.7–11.4 GB。按此最多同时跑两个 scale=1 的中等任务。
- 84/126/98 张和马远 PSD 的第一次运行（任何 scale）一律独占，先拿到峰值。
- 编译也占 CPU 和内存：`cargo build --release` 期间不启动新的运行；同一时刻最多一个 lane 在做 release 编译。
- **计时**：并行运行的耗时在报告里标 `contended: true`，只用于判断能否跑完，不作为性能结论。性能对比（L9、PF.x）只在独占槽里做，并沿用 PF.6 的交错轮次方法。

### 5.3 代码与分支
- **基准分支是 `v1`**（2026-11-12用户确认）。当前基础版本为标签 `base-2026-11-12`；之后每次 L0 合入并冻结的 I(n+1) 都直接落在 `v1` 上。其他历史分支不再作为开发起点：`codex/layered-focus-modularize-2026-10-26` 仅作存档，`codex/accepted-stack-v28-baseline` 仅作 v28 参照，`backup/*` 为备份。

- 每个改代码的 lane 在 `.worktrees/<lane>` 开 worktree，分支 `lane/<lane>/<topic>`，从 L0 最新冻结的集成提交开出，不从别的 lane 分支开。
- 每个 worktree 使用自己的 `CARGO_TARGET_DIR`（放在 worktree 内），避免互相覆盖编译产物。每个约占数 GB 磁盘，计入 §5.5 配额。
- **热点文件**（§5.1 表中列出的文件，以及 `stitch_images.rs`、`stack_pipeline/report_schema_*.rs`）修改前，先在 lane 卡上登记“将修改的函数和原因”；L0 发现两个 lane 登记了同一函数时，安排先后。
- lane 内禁止修改共享质量阈值、全局默认开关和其他素材的基线文件。确需修改时，在 lane 卡上提出，由 L0 汇总后请用户决定。
- 合入只能通过 L0：lane 完成后提交合入请求（lane 卡里写明提交范围、受影响素材、本 lane 验证结果）；L0 rebase 到最新集成提交，跑 §6 回归矩阵，通过后合入并冻结新的集成二进制 I(n+1)，其他 lane 随后 rebase。
- 每次只合入一个 lane，避免回归时无法归因。

### 5.4 数据与输出

- 源素材目录和参考图只读，任何 lane 不得在 `/Users/dp/Downloads/书画拼图` 下写文件。
- 运行输出放在 `output/acceptance/<lane>/<dataset>-<版本>/`，诊断目录放在其下的 `diagnostics/`。不同 lane 不共用输出目录。
- 机器摘要写入 `docs/benchmarks/<dataset>-<版本>-<主题>.json`，至少包含：集成提交和二进制 SHA、输入 manifest 摘要、scale 和开关、实际路径、QG 结果和 ROI 数、失败阶段、来源计数、峰值 RSS、wall、`contended`、输出 TIFF SHA。
- 版本号按 lane 递增（例如 `qixu-1-r1`、`qixu-1-r2`），不和既有 v28–v60 混用，避免覆盖历史。

### 5.5 磁盘

- 每个 lane 的输出和编译产物合计配额 20 GiB；全局剩余低于 40 GiB 时，L0 暂停所有新运行并请用户决定清理范围。
- 原生 TIFF（兰亭约 2.6 GiB）只保留每个等级的通过版本和最近一次失败版本，其他由 lane 在卡上列出后经 L0 清理。不删除 `docs/benchmarks/` 里的任何摘要。

### 5.6 进度记录

- 各 lane 只写自己的 lane 卡 `docs/artwork-lanes/<lane>.md`：当前等级、最近运行、失败阶段、登记的热点修改、待用户决定的问题。
- 只有 L0 修改 `docs/project-progress.md`，按 lane 卡汇总，避免多人同时编辑同一文件。
- 需要用户视觉确认（D4）的结果，由 L0 每天或每批汇总成一次复看清单（整图 + 关键原生裁片路径），不让多个 lane 分别打断用户。

## 6. 回归矩阵

L0 在每次合入前、lane 在提交合入请求前都要跑。

| 层级 | 内容 | 估计耗时（串行） |
| --- | --- | --- |
| 快速 | `cargo fmt --check`、`cargo check`、Rust 库测试（当前 1053 项）、`npm run focus-stack-quality:check`、`focus-stack-datasets:check`、`focus-stack-regression:check`、`focus-stack-lanting:guard:check` | 编译加测试，十几分钟级 |
| 中等 | 9 张冷/热（TIFF 与 v28 一致）；阆苑 10、文苑 10（TIFF 与当前冻结摘要一致）；兰亭 guard `--regression-only`；祁序/枫鹰预览基线 | 约 40–50 分钟串行；两个运行槽并行约 25–30 分钟 |
| 全量 | 已达到 D3 及以上的全量素材 | 随素材增加；S2 后补估计 |

- lane 自己的改动只影响一套素材时，可以先跑快速层和该素材，但合入前中等层必须全过。
- 任一项摘要变化：如果是本次改动的预期效果，在合入请求里逐项说明并附视觉复看；否则拒绝合入。

## 7. S2 全量摸底

S0、S1 完成后立即开始，所有 lane 用同一个 I1 冻结二进制，只跑不改代码。

建议顺序（按内存风险从低到高，前两组可以两两并行）：

1. 马远2 47 张、睡觉 45 张：scale 0.25
2. 祁序1 75 张、枫鹰（三文件夹）：scale 0.25；与分支基线对照
3. 阆苑 84 张：scale 0.25，独占
4. 祁序2 98 张：scale 0.25，独占
5. 文苑 126 张：scale 0.25，独占
6. 马远 PSD 14 个：独占，最后跑

每套产出一张**诊断卡**（写在 lane 卡里）：

- 参与/排除的源图数和排除原因；
- 实际路径（streaming legacy / Virtual_Tile / 其他）和 station 数；
- 停止阶段：特征匹配、关系门禁、不连通分量、闭环、渲染、QG、资源；
- N4.5 失败链计数（无预测重叠、Coverage 不足、双向匹配不足、网格不足、凸包、方向、闭环）；
- 峰值 RSS、wall（标 contended）；
- 预览图和最明显问题的位置；
- 初步判断：属于哪个 lane 的问题，是否需要 L8/L11 的横向能力。

S2 的结论决定 S3 内部的优先级，以及 L9 是否必须提前（例如 126 张在 scale 0.25 就超过内存门）。

## 8. S3–S5 要点

### S3 按素材修复

- 每个 lane 先针对诊断卡里最靠前的停止阶段修复，再往后走，不跳级。
- 修复时优先改证据和测量，不改门槛；必须改门槛时提交给用户决定。
- 一套素材到 D3 后，由 lane 准备复看清单，L0 汇总给用户做 D4。
- D4 通过后跑两次独立运行做 D5，并冻结该素材的回归保护（参考兰亭 guard 的做法），加入 §6 中等层或全量层。

### S4 书画专用能力

- **L8 置换检验**：在同一素材中随机抽取确定不相邻的机位对，用现有匹配和评分算“随机也能达到的相似度”，先只写进 Stack_Report 作为附加证据，与现有固定门槛对照，观察是否能识别重复题字/人物产生的假边。
- **L8 注册 provider**：FAST/BRIEF、SIFT 或 AKAZE、ECC 精修、LightGlue+DISK/ALIKED（ONNX，经 `ort`）、RoMa（只用于连通分量边界）对照，指标见[开源对比 §10.2](focus-stack-open-source-comparison.md)。provider 只提交候选，最终门禁不变。注意许可证：不能用 SuperPoint 权重和 MASt3R。
- **OwnerHybrid**：笔画和印章保持硬 owner，纸面低频用多频带，针对兰亭和枫鹰这类整组曝光不均的接缝。
- **L11 曲率模型**：只在 S2 证明有素材因纸面起伏失败时启动。
- **L10 接缝诊断**：可以和其他 lane 一直并行，用书画真实接痕做标注和负例。

### S5 产品化

- N5.5 来源锁定修图；
- 对齐复核视图和“为什么没对上”的失败说明，直接复用 Stack_Report 和诊断卡的字段；
- N8.2 确定性矩阵（1/2/8/16 线程、改名乱序、取消、内存超限）；
- 桌面完整流程（ENG.3）和发布（ENG.4）。

## 9. 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| S0 合并冲突改变兰亭或 9 张输出 | 所有 lane 推迟 | 冲突按提交逐个归因；无法两全时交用户决定 |
| 84/126 张超过 32 GiB 内存 | 无法到 D2 | S2 先跑 scale 0.25 拿峰值；超限则 L9 提前，优先做按需 tile（N6.4） |
| 阆苑输出 P95 余量只有约 0.012px | 扩到 84 张后可能不过 | L2 先看全量里的最差边，不调门槛 |
| 无参考图的素材（祁序、马远2、睡觉、文苑） | D4 全靠用户看 | 统一复看清单，按批次请用户确认 |
| 多个 lane 同时改流式 mosaic | 合入冲突、回归无法归因 | 热点登记 + 一次只合入一个 lane |
| 并行运行导致计时失真 | 性能结论不可信 | `contended` 标记；性能只在独占槽测 |
| 用户复看速度跟不上 agent 产出 | 大量结果堆积在 D3 | L0 控制同时处于 D3 待看的素材数（建议不超过 3 套） |

## 10. 给 lane agent 的任务说明模板

启动每个 lane 时，在 brief 里至少写明：

```text
Lane: <L编号 名称>
目标: <把 <dataset> 从 D<n> 推进到 D<n+1>，或具体任务>
基础: 集成提交 <sha>，冻结二进制 <路径> (SHA-256 <...>)
工作区: /Users/dp/Desktop/raw-editor/.worktrees/<lane>，分支 lane/<lane>/<topic>
CARGO_TARGET_DIR: /Users/dp/Desktop/raw-editor/.worktrees/<lane>/target
素材: /Users/dp/Downloads/书画拼图/<目录>（只读）
输出: /Users/dp/Desktop/raw-editor/output/acceptance/<lane>/<dataset>-<版本>/
运行槽: 运行前在 output/acceptance/.run-slots/ 登记，预估峰值 <GiB>
已知情况: <诊断卡要点、历史版本、相关 benchmark 路径>
禁止: 修改共享阈值/默认开关/其他素材基线；写入素材目录；直接合入 v1；修改 docs/project-progress.md
热点文件: <本 lane 可能修改的文件>；修改前在 docs/artwork-lanes/<lane>.md 登记
完成时提交: lane 卡更新、机器摘要 docs/benchmarks/<...>.json、§6 回归矩阵结果、需用户确认的问题
```

## 11. 新增或调整的任务编号

在总览中对应如下：

- **ENG.6**：S0 两条代码线集成（新增）。
- **N8.5**：并行运行基础设施，即 §5 的规则落地（新增）。
- **N7.5**：书画交付等级 D1–D5（新增）。
- **N3.5**：按 §7 执行全量摸底，并按 lane 推进。
- **PF.8**：增加置换检验子项。
- **PF.10**：融合部分缩为 OwnerHybrid，通用融合基线降为离线参考。
- **N9.1**：从后续研究升为 S4 正式任务，启动条件见 §8。
- **BR.4**：通用场景验收降级，不进入当前顺序。
