# 任务14：文苑126张到 D2/D3 诊断边界

- 更新：2026-11-18；状态：入口/密度/tone 审计完成，原生推进未开始；分支 `v1`。
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

只读对比显示：0.25 使用 `supported_spatial_affine`，4 个 tile 被 clamp，source-fidelity strength 最小值 `1.0`；scale=1 切换为 `station_affine`，32 个 tile 被 clamp，source-fidelity strength 最小值降至 `0.111686`。对应 boundary ΔE00 从 `4.6045643/13` 变为 `8.7789013/142`。这说明 scale1 的 tone 模式/原生 source retention 是下一项实验入口；不放宽 1.5 阈值，也不直接启用生产 fallback。

机器结果：[wenyuan-tone-mode-audit.json](/Users/dp/Desktop/raw-editor/output/acceptance/task-14/wenyuan-tone-mode-audit.json)。

## test-only tone source-fidelity 假设

为验证 scale=1 的 `all_have_native_sources=false` 是否只是把 tone 路径切到 `station_affine`，加入了仅在 Rust library test binary 编译的开关 `RAW_EDITOR_TEST_FORCE_TONE_SOURCE_FIDELITY=1`。它只把本次调用传给 tone harmonizer 的 fidelity 布尔值强制为 true；生产构建没有该路径，默认行为不变，也不放宽 tone threshold 或启用 fallback。

本次先完成 `cargo check --tests --offline` 和 `git diff --check`，未把编译通过当作质量改善证据。完整 scale=1 A/B 仍需独占长跑；后续先用 Wenyuan-10 bounded run 检查 tone mode、owner/clamp 与 boundary 输出，再决定是否值得消耗完整 126 输入的运行预算。

bounded 结果已落到 [wenyuan-tone-source-fidelity-r1.json](../benchmarks/wenyuan-tone-source-fidelity-r1.json)：0.25 与 scale=1 的 Wenyuan-10 都没有 excluded source，实际仍走 `supported_spatial_affine`，没有触发强制开关的分支；scale=1 子集完整通过 native acceptance，ΔE00 最大 `0.177211`、0 个 tone 违规、density `applied`，内存峰值 `11837079552` bytes。0.25 子集仍因 `local_scale_median:not_applicable` 被拒绝。这个结果只能证明 bounded 合同和 hook 不破坏短跑，不能证明它改善了完整 126 输入 scale=1 的 mode shift；完整 A/B 仍待独占运行。
