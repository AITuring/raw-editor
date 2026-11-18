# 任务14：文苑126张到 D2/D3 诊断边界

- 更新：2026-11-18；状态：入口/密度/tone 审计完成，原生推进未开始；分支 `lane/task-14/wenyuan-diagnostic`。
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

任务14不跑 scale=1，也不修改 `expectedSkipped` 来制造来源通过。下一步先把空源文件登记/排除合同、density bypass 入口和 tone 违规拆成可验证的单变量 test-only 假设；完整 QG 与入口合同同时闭合后，才考虑 native。
