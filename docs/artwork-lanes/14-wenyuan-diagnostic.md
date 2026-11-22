# 任务14：文苑126张到 D2/D3 诊断边界

- 更新：2026-11-23；状态：入口/density/tone 根因诊断收口，D3/native 未通过，交接任务16；任务13已完成 residual/scale/boundary 根因诊断并交接；分支 `v1`。
- 输入：文苑图全量126张，scale=0.25；沿用任务5冻结 `base-2026-11-12`，不改生产代码、阈值、矩阵或正式基线。
- 依据：任务5输出 `/Users/dp/Desktop/raw-editor/output/acceptance/task-5/wenyuan-baseline-r1` 及 [S2-wenyuan](S2-wenyuan.md)。

## 入口与 QG

本轮报告选中 `layered_virtual_tile`，closure `converged`，QG 汇总为 pass；但 native/runner 仍拒绝，不能把 QG 汇总当作 D3。126 个登记输入中 121 张为 station fusion、4 张 singleton tile，`DSC_4885.NEF` 在启动时就是 0 bytes/空 SHA，native 明确 `File is empty` 并排除。manifest 的 `expectedSkipped` 仍为空，`report_source_digests=false`，入口合同未闭合。

QG 的 local-scale、effective、gradient、owner 等有效门均 pass 或 not_applicable，unresolved 为 0；这不覆盖空源入口拒绝和 tone degraded。

## density / tone 核查

- output density：`bypassed`，factor `1.0`、sample_count `0`，原因 `nonidentity_world_residual`；不能据此推进 scale=1 或宣称 native 密度安全；
- tone：`degraded`，boundary ΔE00 最大 `4.6045642866 > 1.5`，13 条违规；最坏区域为 station5↔32，world rect `(5888,1280,256,256)`，另有 station37↔35 约 `(4096,3072,256,256)`、ΔE `3.2804`；
- 资源：峰值 RSS `15111585792` bytes（14.07 GiB），Rayon=8、预处理=6，wall `9558.56s`，融合230次、RAW decode1207次；无 task15 硬触发条件。

机器摘要：[wenyuan-task-14-diagnostic.json](../benchmarks/wenyuan-task-14-diagnostic.json)。

## 结论与下一步

任务14不跑 scale=1，也不修改 `expectedSkipped` 来制造来源通过。空源排除合同与 density/tone 只读审计已经完成；当前仍没有完整入口合同、原生 density 采样或 tone 通过证据。下一步只能选择一个可复现的 test-only 单变量假设，先在同一 0.25 范围闭合入口合同和完整 QG，再决定是否推进 native。

## 空源 shadow contract

使用 [wenyuan_contract_audit.py](../../scripts/stack-lab/wenyuan_contract_audit.py) 做只读 shadow manifest 审计：若将 `DSC_4885.NEF` 加入 expectedSkipped，能表达“预期排除”，但当前 Stack_Report 仍包含该 source 记录（`sha256_before=""`、`decoded=false`），因此仅改 expectedSkipped 也不足以让 `report_source_digests` 通过；需要单独设计并评审“排除源不进入 report.sources”的入口合同，再重新生成 baseline。正式 manifest、sourceAggregateSha256 和历史基线均未修改。

机器结果：`/Users/dp/Desktop/raw-editor/output/acceptance/task-14/wenyuan-empty-contract-audit.json`。

## 排除源 report shadow contract（test-only）

脚本 [wenyuan_exclusion_contract_audit.py](../../scripts/stack-lab/wenyuan_exclusion_contract_audit.py) 对任务5冻结报告同时检查严格合同与候选合同。严格合同仍失败：报告含 126 条 source 记录而活动输入只有 125 条，`report_source_digests=false`。候选合同把 `DSC_4885.NEF` 从活动 source path equality 中排除，但要求保留明确记录：`decoded=false`、`sha256_before` 为空、`sha256_after` 等于该空文件的不可变 SHA；基线审计通过（活动源路径/摘要均匹配，排除记录完整）。

这只是入口设计证据，不是正式验收，也没有写入 manifest、runner 或历史基线。若要采用，必须让 report schema、路径集合检查和重生成 baseline 一起变更，并继续单独处理 tone degraded 与 density bypass。

机器结果：[wenyuan-exclusion-contract-audit.json](/Users/dp/Desktop/raw-editor/output/acceptance/task-14/wenyuan-exclusion-contract-audit.json)。

## tone / density 只读审计

脚本 [wenyuan_tone_density_audit.py](../../scripts/stack-lab/wenyuan_tone_density_audit.py) 对冻结 Stack_Report 与源码标记做只读核查：tone 为 `degraded`，13 个 boundary ΔE 违规，最大 `4.6045642866`（station 5↔32，world `(5888,1280,256,256)`）；另有 4 个 `tone_gain_clamped` station（34、36、38、39）。output density 为 `bypassed`、factor `1.0`、`sample_count=0`，原因 `nonidentity_world_residual`，因此没有 native Jacobian 密度证据。源码检查只找到 cfg(test) capture/no-op 标记，没有 tone 或 density 数值覆盖开关。

机器结果：[wenyuan-tone-density-audit.json](/Users/dp/Desktop/raw-editor/output/acceptance/task-14/wenyuan-tone-density-audit.json)。

## tone owner oracle A/B（Wenyuan-10，scale=0.25）

在冻结 `base-2026-11-12` 上只开启现有 `RAW_EDITOR_TONE_OWNER_FIDELITY_DIR` 与 `RAW_EDITOR_TONE_DIAGNOSTICS_DIR` sink，运行 `wenyuan-10` 子集。运行本身保持 `report_source_digests=true`、QG `pass`、内存门通过（峰值 `10260135936` bytes），但 native acceptance 因 `local_scale_median` 的 not_applicable 判定退出（child exit `101`），所以这是诊断证据而非接受结果。

owner oracle 的 bounded shared-world-field relaxation 没有联合可行解：边界要达到门槛至少需要 relaxation `0.743408203125`，owner-fidelity 的最大可行 relaxation 是 `0.0`，Pareto gap 为 `0.743408203125`，最终回退 identity。这个单变量 tone relaxation 假设被否定；不能把它升级为生产 fallback。`tone-final-world` sink 在自身 24 个窗口上测得最大 ΔE00 `0.2666573`，但它不替代 formal owner oracle/QG 判定。

机器结果：[wenyuan-tone-oracle-audit.json](/Users/dp/Desktop/raw-editor/output/acceptance/task-14/wenyuan-tone-oracle-audit.json)。

## 空源恢复核查

只读搜索 `/Users/dp/Downloads`、`/Users/dp/Desktop` 和 `/private/tmp` 后，`DSC_4885.NEF` 只有一个本地副本：0 bytes，SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。没有可用于补源或重生成的非空副本；不能伪造源文件或据此重基线。

## 排除源契约设计稿

已将 shadow contract 固化为 [14-wenyuan-excluded-source-contract.md](14-wenyuan-excluded-source-contract.md)：活动 `sources` 与 `excluded_sources` 分离，保留空源的 decode/摘要证据，并要求选中、活动、排除三项计数闭合。该设计仍待 L0 评审和正式 report/runner 实现；不改变当前入口判定。

## schema-8 + expectedSkipped 全量验证（r1）

使用冻结二进制 `task-14-excluded-source-r1`（SHA `0c353860b134f216aedbc2aaf3f6ca5f809d186e1c713bc3a7ce89e8bb236971`）和 manifest `expectedSkipped=["DSC_4885.NEF"]` 完成 Wenyuan 126 输入、scale=0.25 独占运行。来源合同已闭合：`report_source_digests=true`；Stack_Report 保留 126 条原始 `sources`，并新增 1 条 `input.excluded_sources`，`decoded=false`、`sha256_before=""`、`sha256_after` 等于空文件 SHA。活动输入 125 张，ordered 125 张。

运行状态仍为 `rejected`，native child exit `101`，原因是既有 `local_scale_median:not_applicable`；QG 汇总 `pass`、TIFF 写出和 24GiB 内存门通过。tone 仍为 `degraded`，边界 ΔE00 最大 `4.6045642866`；density 仍为 `bypassed`、`sample_count=0`、`nonidentity_world_residual`。这次运行只证明排除源入口合同，不授予 D2/D3。

机器结果：[wenyuan-task-14-excluded-source-r1.json](../benchmarks/wenyuan-task-14-excluded-source-r1.json)。

## scale=1 native 验证（r1）

在入口契约闭合后，用同一 schema-8 冻结二进制 `task-14-excluded-source-r1` 完成 Wenyuan 126 输入 scale=1 独占运行。native 几何质量门首次进入真实尺度：`local_scale_median=pass`、`local_scale_pixel_ratio=pass`、closure `converged`；QG 汇总 `pass`，TIFF 为 `24681×13091` RGBA16，内存门通过（峰值 `12939411456` bytes，wall `17192.31s`）。

native 仍拒绝，唯一 runner failure 为 tone boundary：最大 ΔE00 `8.778901254993071`，142 个违规；density 仍因 `nonidentity_world_residual` bypass，`sample_count=0`，没有 native 密度证明。来源合同继续通过（`report_source_digests=true`，1 条 excluded source）。本次没有超 24GiB 或跑不完证据，不启动任务15。

机器结果：[wenyuan-task-14-scale1-r1.json](../benchmarks/wenyuan-task-14-scale1-r1.json)。

## scale1 tone mode audit

只读对比显示：0.25 使用 `supported_spatial_affine`，4 个 tile 被 clamp，source-fidelity strength 最小值 `1.0`；scale=1 切换为 `station_affine`，32 个 tile 被 clamp，source-fidelity strength 最小值降至 `0.111686`。对应 boundary ΔE00 从 `4.6045643/13` 变为 `8.7789013/142`。后续通过源码和现有 report 锁定为 local fit 的 40M allocation guard；不放宽 1.5 阈值，也不直接启用生产 fallback。

机器结果：[wenyuan-tone-mode-audit.json](/Users/dp/Desktop/raw-editor/output/acceptance/task-14/wenyuan-tone-mode-audit.json)。

## test-only tone source-fidelity 假设（已关闭）

为验证 scale=1 的 `all_have_native_sources` 是否只是把 tone 路径切到 `station_affine`，短暂加入了仅在 Rust library test binary 编译的开关 `RAW_EDITOR_TEST_FORCE_TONE_SOURCE_FIDELITY=1`。它通过编译检查并完成 bounded/full 实验；实验结束后已移除，生产默认和入口合同没有改变。

本次没有把编译通过当作质量改善证据，也没有放宽 tone threshold 或启用 fallback。完整 A/B 证明 baseline 本身已经是 `source_fidelity=true`，强制开关只是 `true→true` 的冗余对照，因此不保留这个无效开关。

bounded 结果已落到 [wenyuan-tone-source-fidelity-r1.json](../benchmarks/wenyuan-tone-source-fidelity-r1.json)：0.25 与 scale=1 的 Wenyuan-10 都没有 excluded source，实际仍走 `supported_spatial_affine`；scale=1 子集完整通过 native acceptance，ΔE00 最大 `0.177211`、0 个 tone 违规、density `applied`，内存峰值 `11837079552` bytes。0.25 子集仍因 `local_scale_median:not_applicable` 被拒绝。这个结果只能证明 bounded 合同和 hook 不破坏短跑，不能单独解释完整 126 输入 scale=1 的 mode shift；完整 A/B 结论见下段。

完整 A/B 已完成：强制开关运行 126 输入 scale=1 独占 `17154.48s`，内存峰值 `16153575424` bytes，QG 和内存门通过，但 native 仍因 tone boundary 最大 ΔE00 `8.778901`、142 个违规而拒绝；`correction_model=station_affine`、32 个 tile clamp、最小 source-fidelity strength `0.111686`，density 仍为 `bypassed/nonidentity_world_residual`。输出 SHA 与无开关基线完全相同（`f253dddc…`），说明 baseline 本身已经进入 source-fidelity 分支，这个强制布尔值只是 `true→true` 的冗余对照；mode shift 的决定性原因是 local fit 的 40M allocation guard，不启用生产 fallback，也不放宽 tone threshold。

机器结果：[wenyuan-tone-source-fidelity-r1.json](../benchmarks/wenyuan-tone-source-fidelity-r1.json)。下一步回到诊断：local fit 的 40M allocation guard 已由现有 report 和公式锁定；已加入 test-only `tone-local-fit.json` sink，未来启用 tone diagnostics 时记录 guard reason/grid/product。本次没有为生成该 sink 再跑 full run；默认上限和生产路径不变，不再重复同一 full A/B。公式审计见 [wenyuan-local-tone-guard-audit.json](../benchmarks/wenyuan-local-tone-guard-audit.json)。

## tone boundary 残差 bounded A/B（r1）

当前 `task-14-tone-adaptive-r1` 的 52 个 tone boundary 违例集中在右下 station 34–39，主峰是 36↔37（最大 ΔE00 `5.447763`）。这些 tile 的局部 gain/offset 范围触及现有边界，但对应关系本身仍有较高几何残差，因此不能直接把问题归因于 bounds。

先用同一批已保存的 Wenyuan-10 native tone fields 做单变量 test-only A/B：控制组使用 gain/offset bound scale `1.0`，实验组只将该 scale 改为 `1.5`。两组最大 ΔE00 都是 `0.085911`、违规窗口都是 `0`，参数范围没有实质变化，说明这个 bounded subset 没有命中边界，实验对 full 34–39 cluster 不具判别力。生产 bounds、1.5 阈值和 fallback 均保持不变。

同一机器摘要还保留了 full scale 的前后对照：`station_affine` 的 `8.778901/142` 降至 adaptive `supported_spatial_affine` 的 `5.447763/52`，但仍被 native tone boundary 拒绝；这两份 full report 的冻结 commit 不同，只作上下文证据，不当作严格单变量因果 A/B。来源合同、closure、QG 汇总和内存门均通过，density 仍是 `bypassed/nonidentity_world_residual`。因此本轮不升级生产参数，也不重复 5 小时全量运行；下一步先导出 full cluster 的 test-only tone fields，或建立一个明确命中 bounds 的 synthetic fixture，再决定是否值得新一轮验证。

机器摘要：[wenyuan-tone-boundary-ab-r1.json](../benchmarks/wenyuan-tone-boundary-ab-r1.json)。

## 34–39 synthetic bound-hit fixture

冻结的 full binary 没有导出 34–39 的真实 local `ToneTile` 字段；现有 `capture_test_tone_fields` 只在 `cfg(test)` sink 中生效，无法从当前 Stack_Report 反推完整字段。因此补了一个只在 test binary 中运行的六站 fixture，站点编号固定为 34–39、使用六段垂直 owner band 和链式 tone relation，专门验证 solver 在边界命中时的响应。

控制组使用生产 bounds `[0.8, 1.25]`、`±0.02`，ΔE00 从 `7.915623` 降到 `3.840939`；实验组只将 test-only bound scale 改为 `1.5`，bounds 变为 `[0.7, 1.375]`、`±0.03`，ΔE00 降到 `0.482869`。两组都明确命中 bounds，说明 solver 对该变量有响应；这不能证明真实 34–39 残差由 bounds 主导，所以不升级生产参数、不放宽正式阈值，也不启动全量运行。

机器摘要：[wenyuan-tone-boundary-synthetic-ab-r1.json](../benchmarks/wenyuan-tone-boundary-synthetic-ab-r1.json)。下一步使用真实 34–39 小范围字段做离线 boundary replay，再决定是否需要新的 bounded 实验。

## 真实 34–39 小范围 tone field capture

已从任务14 full Stack_Report 的 grouping 反查出 34–39 的 15 个真实 NEF，并只用这 15 个源启动 ignored Rust test fixture，开启 `RAW_EDITOR_TONE_DIAGNOSTICS_DIR` 和 preview-only。capture sink 现在支持按 run-local station index 的 test-only 过滤，并能写出过滤后的 owner plane 与 manifest；本次小范围验证保留了全部 6 个 run-local tile，写出 6 组真实 `tone-owner-*.json` 与 `.native.f32`，总 native field `198,348,672` bytes。由于小范围重分组，输出 station index 是 `0–5`，原始 34–39 的归属由 source subset manifest 保留；若在 full run 使用原始 station index，则再使用 `RAW_EDITOR_TEST_TONE_STATIONS=34,35,36,37,38,39`。

字段对、过滤后的 owner plane 和 native proof 已在同一 15 源小范围的 r7 capture 中完成：6/6 组 `tone-owner-*.json` 与 `.native.f32`、`tone-owners.filtered.png`（16134×6170）、34–39 标签映射，以及 13 条带 `native_verified_points` 的 authoritative relation。capture manifest 的 `missing_stations` 为 `[]`，字段审计通过；saved replay loader 也支持从 filtered owner plane 回退读取。机器摘要：[wenyuan-tone-fields-capture-r2.json](../benchmarks/wenyuan-tone-fields-capture-r2.json)。

随后以 r7 的真实 owner plane、字段和 `joint-production-input.json` 运行 `real_saved_local_tone_from_env`。replay 没有通过：13 条关系拟合 1000 次后，最大 tone prediction Δ 为 `10.857647310904358`，输出保存在 `/private/tmp/wenyuan-native-proof-34-39-r7/saved-local-tone-boundary.json`。这证明保存的真实 34–39 字段仍存在严重的 boundary 残差，但不说明应放宽生产 bounds 或 1.5 阈值。capture-only harness 没有正式 QG、source contract 或 memory-gate 输出；日志中的 density 仍是 `bypassed/nonidentity_world_residual`，因此不作这些门的通过声明，也不外推到 126 源 full run。

在同一保存字段上做了单变量 bounded A/B：控制组 `RAW_EDITOR_TEST_TONE_LOCAL_BOUND_SCALE=1.0`，实验组仅改为 `1.5`。最大残差由 `10.857647310904358` 降至 `10.457657219451127`，失败窗口由 `81` 降至 `73`，约改善 `3.68%`；参数范围确实从 gain `[0.8,1.25]`、offset `±0.02` 扩到 gain `[0.7,1.375]`、offset `±0.03`，但两组都仍失败。因此 bounds 对该真实残差有轻微响应，却不是充分修复；生产 bounds、阈值和 fallback 保持不变。机器摘要：[wenyuan-tone-fields-replay-ab-r1.json](../benchmarks/wenyuan-tone-fields-replay-ab-r1.json)。

残差定位显示，最大两个 witness 是 36↔38（world `(5888,5632)`，Δ `10.857647`→`10.457657`）和 36↔37（`(7680,5632)`，Δ `9.602754`→`9.336838`），两者控制组参数都没有触及 bounds；37↔38 才是明确 bound-hit，并从 `7.6763738` 降到 `3.9173246`。因此下一步应隔离 36↔38、36↔37 的 source/Jacobian 或 relation-model 证据，不能继续把扩大 bounds 当作主修复路径。

对保存的低频字段做了只读定位：36↔38 的最大 witness 两侧低频亮度比约 `37.99×`，两侧 validity/coverage 都是有效；对应 native relation 有 `308` 个 verified points、误差 `1.4985px`。36↔37 也有 `408` 个 verified points、误差 `1.4901px`，但仍保持约 `9.60` 的边界残差。两个主 witness 在左右两侧都落在 native verified correspondence 的轴对齐支持框之外，说明当前边界值依赖 relation/source-Jacobian 外推；这个组合更像 source/Jacobian 映射或关系模型与亮度场不一致，而不是单纯 affine box 太窄。机器摘要：[wenyuan-tone-fields-residual-localization-r1.json](../benchmarks/wenyuan-tone-fields-residual-localization-r1.json)。

进一步做了 support-aware 分类：控制组 81/81 个失败窗口、实验组 73/73 个失败窗口都落在 native correspondence 支持框之外，支持框内失败数均为 0；控制组仍有 7 个 bound-hit，实验组降为 2 个。该分类使用两侧 native verified 点的轴对齐 bbox，只是保守支持代理，不是几何通过证据；它把下一步收敛到 36↔38、36↔37 的 boundary-local source/Jacobian 支持验证。机器摘要：[wenyuan-tone-fields-support-audit-r1.json](../benchmarks/wenyuan-tone-fields-support-audit-r1.json)。

进一步量化支持缺口：36↔38 的两个 witness 距左右 native correspondence bbox 分别约 `2364px` 和 `795px`，36↔37 分别约 `2423px` 和 `1675px`，方向均指向右下边界；但对应 field cells 的 validity/coverage 仍是 `255/1.0`。这说明字段本身有值，缺的是该边界位置的 native verified relation 支持，而不是简单的空洞或 bounds 饱和。机器摘要：[wenyuan-tone-fields-support-gap-r1.json](../benchmarks/wenyuan-tone-fields-support-gap-r1.json)。

随后做了一个独立 source-resampling 路径核对：只对 `DSC_4862.NEF` 请求 divisor `1.25`，并在 test binary 中强制尝试该 hook。r8/r9 与 r7 的 6 组 field、owner plane、13 条 native proof 和 saved replay 均逐文件一致（replay 最大 Δ 都是 `10.857647310904358`）。代码路径核对显示当前 `FocusStack→LayeredVirtualTile` 组合不经过 `streaming_mosaic` 的 resampling hook，因此这不是可判别的 source/Jacobian 实验；不再重复该 hook，下一步只在 active LayeredVirtualTile 路径补 boundary-local native correspondence/source sampling 证据。机器摘要：[wenyuan-tone-fields-source-resample-hook-r1.json](../benchmarks/wenyuan-tone-fields-source-resample-hook-r1.json)。

active LayeredVirtualTile 的第一版 `native_point_trace` 使用了错误的 `reference_canvas_origin=[0,0]`，其“全部 outside / 无 sampling”结论已标记为 superseded，不能用于算法判断：[wenyuan-tone-fields-active-point-trace-r1.json](../benchmarks/wenyuan-tone-fields-active-point-trace-r1.json)。

用 active `world_origin=[-7724,4]` 校正后的 r2 trace 显示：18 行中 13 行进入 station bbox，9 行执行 source diagnostics，8 行完成 native owner resolution，所有报告的 source footprint/global support 均为 true，最终 3 个 witness 都有 coverage。修正后的 field/bbox 对照为 saved field 覆盖 15/18、active tile 覆盖 11/18；r1 的“坐标域完全不一致”同样由 origin 错配造成。机器摘要：[wenyuan-tone-fields-active-point-trace-r2.json](../benchmarks/wenyuan-tone-fields-active-point-trace-r2.json)、[wenyuan-tone-fields-active-field-bbox-r2.json](../benchmarks/wenyuan-tone-fields-active-field-bbox-r2.json)。

因此当前结论改为：active source sampling 存在且多数 witness 可达，剩余问题应比较 active owner/source 的实际采样 RGB 与 saved tone relation，而不是继续扩大 bounds 或把问题归因于普遍的 source support 缺失。

对照结果已落盘：36↔38 的 active 左侧 owner 没有被选中，而右侧 `DSC_4864.NEF` 有效采样；这是一项 owner-selection gap。36↔37 的 `DSC_4867.NEF` 与 `DSC_4848.NEF` 都完成了无 fallback 的有效采样，但亮度比约 `100,776×`；37↔38 的 `DSC_4848.NEF` 与 `DSC_4861.NEF` 亮度比约 `111.5×`。因此 36↔37、37↔38 是 valid source sampling 下的直接 photometric mismatch，36↔38 还叠加 owner 未选中；下一步应分别处理 owner 选择与 relation photometric mismatch，不再统一扩大 bounds。机器摘要：[wenyuan-active-photometric-mismatch-r1.json](../benchmarks/wenyuan-active-photometric-mismatch-r1.json)。

## owner revisit 与 relation model 修正

在 active LayeredVirtualTile 路径补了一个有界 owner guard：native ownership revisit 不再用近黑/前景候选覆盖已经选定且亮度可支持的画布 owner；初始 owner 与 station seam revisit 继续使用同一判定。单元测试覆盖“亮画布 owner + 近黑候选”和反向替换两种方向，生产 tone bounds、正式阈值和 fallback 均未改动。

使用同一 corrected r2 witness 输入和 15 张真实 NEF 重跑 r5 后，三个 witness 都有 coverage。36↔37 的最终 owner 从旧 r2 的 station 3（station 37，RGB 约 `[0, 7.8e-8, 2.2e-5]`）变为 station 4（station 36，RGB `[0.20177, 0.15467, 0.16129]`）；37↔38 保持 station 5，36↔38 保持 station 6。这个变化证明 revisit 覆盖问题已修正，但它不是 full-run QG 通过：r5 Stack_Report 的 tone status 仍为 `degraded`，boundary max ΔE00 `9.420790`（阈值 `1.5`，95 个窗口超限），Quality_Gate `fail`。

同一 r5 owner plane 的 saved-native replay 在生产 bounds scale `1.0` 下为 max ΔE00 `9.420790`、95 个失败窗口；test-only 将 relation luminance floor 从 `0.02` 降至 `0.001` 后为 `7.457318`、76 个失败窗口，仍失败。因此低亮度 relation 样本是有响应的模型变量，但当前证据不足以把它提升为生产默认；它保持 `cfg(test)`，不以扩大 bounds 代替 source/Jacobian 或 pairwise relation 修复。r5 的 source preflight 为 15/15 stat、file-map、hash、EXIF 且失败数 0，peak RSS `12,858,425,344` bytes，memory threshold 未超限；density 仍是 `bypassed/nonidentity_world_residual`，所以不作正式 acceptance 声明。机器摘要：[wenyuan-owner-selection-relation-fix-r1.json](../benchmarks/wenyuan-owner-selection-relation-fix-r1.json)。

基于 r10 saved fields 做了 pairwise relation/source-Jacobian 隔离：36↔37 的 relation 内部 post ΔE00 为 `0.0761`、保留 `86,139` 个样本；active source sampling 两侧均 observed/support=true、无 fallback，但低频亮度比约 `100,776×`。37↔38 的 relation post ΔE00 为 `0.4120`、保留 `52,973` 个样本；active source sampling 同样有效，低频亮度比约 `111.5×`。这两对的 residual 更符合有效 source sampling 下的直接 photometric mismatch，而不是普遍的 source/Jacobian 缺失；luma floor `0.001` 只能把 replay max 从 `9.420790`→`6.822006`、`7.805110`→`6.608655`，仍未通过。36↔38 的亮度比约 `24.23×`、relation post ΔE00 `0.2244`，属于较低强度的同类残差。机器摘要：[wenyuan-tone-fields-pairwise-relation-source-audit-r1.json](../benchmarks/wenyuan-tone-fields-pairwise-relation-source-audit-r1.json)。

对 active source 的亮度比做了 bounded source-normalization brief。以生产 gain `[0.8,1.25]`、offset `±0.02` 和目标 ratio `1.25` 计算，36↔37 需要把亮源 gain 降到 `1.24e-5` 或暗源 gain 提到 `80620.6`；37↔38 需要 `0.0112` 或 `89.21`；36↔38 需要 `0.0516` 或 `19.38`。即使同时使用两侧最有利的 gain/offset 极值，三对的 ratio 下限仍约为 `5.60/4.31/3.59`，且还需额外 offset `0.07–0.11` 才能触及目标。因此 bounded source-specific A/B 没有判别价值；若继续，只能另立有明确物理依据的 source calibration 或 relation-model 变量，不能借此扩大生产 tone bounds。机器摘要：[wenyuan-source-normalization-brief-r1.json](../benchmarks/wenyuan-source-normalization-brief-r1.json)。

随后增加了 test-only `RAW_EDITOR_TEST_TONE_MAX_PAIR_LUMA_RATIO`，在生产 bounds scale `1.0` 下做 pairwise gate A/B：ratio `4` 只把 replay max 从 `9.420790` 降到 `9.401567`，失败窗口仍为 `95`；ratio `16` 与 control 完全相同。该 gate 不能修复边界残差，保留作诊断开关，不提升为生产 relation 规则。机器摘要：[wenyuan-tone-pair-luma-ratio-ab-r1.json](../benchmarks/wenyuan-tone-pair-luma-ratio-ab-r1.json)。

最后对 retained-domain 外的 local tone 参数扩展做了 test-only 半径 A/B：control 保留生产序列 `4/8/16/32/64px`，radius `16` 与 control 完全相同（max ΔE00 `9.420790`、95 个失败窗口），禁用扩展的 radius `0` 反而恶化为 `10.094219`、106 个失败窗口。结论是扩展对当前 replay 是必要的，但不是残差根因；停止继续扫 extension radius，生产扩展逻辑与 bounds 保持不变。机器摘要：[wenyuan-tone-support-extrapolation-ab-r1.json](../benchmarks/wenyuan-tone-support-extrapolation-ab-r1.json)。

基于 active witness 做了 source-specific normalization brief：生产 gain 两侧最多提供 `1.25/0.8=1.5625×` 相对校正，但 36↔38、37↔38、36↔37 仍分别需要约 `15.5×`、`71.4×`、`64,496×` 的额外亮度压缩。这个量级不支持加入 source normalization；它更符合 source/content identity mismatch，而不是统一曝光偏移。机器摘要：[wenyuan-active-source-normalization-brief-r1.json](../benchmarks/wenyuan-active-source-normalization-brief-r1.json)。

对三个 active witness 做了 source identity 审计：36↔37、37↔38、36↔38 两侧实际采样的 NEF 均属于各自预期 station cluster，source sampling 均 observed、无 fallback，也没有跨 station 串源。因而当前证据排除了“source identity/cluster 选错”这一类根因，剩余问题是这些 source 在 witness 位置的内容/亮度对应关系本身。机器摘要：[wenyuan-source-identity-audit-r1.json](../benchmarks/wenyuan-source-identity-audit-r1.json)。

最后导出了五个 witness 所需 source 的 prepared 灰度图（scale `0.5`），按 active trace 的 prepared source 坐标计算半径 `32px/96px` patch。36↔37 的 patch 平均亮度比为 `89.0×/62.6×`，归一化相关为 `-0.044/-0.064`；37↔38 为 `49.3×/46.7×`，相关 `0.002/-0.027`。两对在两个空间尺度都没有可用内容相关性，不能用单一 photometric relation 或 source normalization 解释。36↔38 的亮度比较低（`1.50×/1.43×`），但相关也只有 `0.081/0.279`，属于较低强度的内容/配准 mismatch。机器摘要：[wenyuan-source-content-patch-audit-r1.json](../benchmarks/wenyuan-source-content-patch-audit-r1.json)。

任务14本轮最终结论：owner revisit guard 已修正 36↔37 的错误 owner；source identity、support extrapolation、pair luminance gate 和 bounded source normalization 均不能解释或消除残差；36↔37、37↔38 是 witness-local content/photometric mismatch，36↔38 是较弱的 content/registration mismatch。保留现有 owner guard 与 test-only diagnostics，不修改生产 tone bounds、正式阈值或 fallback；在得到 content correspondence/局部几何修正证据前，不授予 native acceptance。

根据该结论又做了 translation-only local registration search：在 prepared patch 上搜索 ±128 native px，步长 8px。最佳相关仍只有 36↔37 `0.256`、37↔38 `0.110`、36↔38 `0.290`，无法恢复强内容对应；因此小范围平移也不能解释残差。本轮到此收口：不再添加局部平移、source normalization 或 tone bounds 调整；任务14保持 diagnostic rejection，后续若继续只能设计 content-aware correspondence 模型并重新建立独立证据。机器摘要：[wenyuan-source-content-local-registration-search-r1.json](../benchmarks/wenyuan-source-content-local-registration-search-r1.json)。
## 入口、density、tone 根因诊断收口

入口：`DSC_4885.NEF` 是物理缺失的 0-byte 源；schema-8 `excluded_sources` 合同已闭合，但不产生可用图像。density：0.25 与 scale=1 都因 `nonidentity_world_residual` bypass，`sample_count=0`，没有 native Jacobian 密度证明。tone：scale=1 full run 的几何/QG 和 memory gate 通过，但 tone boundary 最大 ΔE00 `8.778901`、142 个窗口超限；34–39 test-only replay 仍为 `9.420790/95`。

owner revisit 已修正 36↔37，source identity 已排除跨 station 串源；剩余证据收敛到 36↔37、37↔38 的 witness-local content/photometric mismatch，以及较弱的 36↔38 content/registration mismatch。任务14到此收口为 diagnostic rejection：不修改 tone bounds、正式阈值、source normalization、fallback 或 pose，不重复 full 126 长跑，后续切换任务16既有 boundary candidate/product-decision lane。

机器摘要：[`wenyuan-task-14-root-cause-closure-r1.json`](../benchmarks/wenyuan-task-14-root-cause-closure-r1.json)。
