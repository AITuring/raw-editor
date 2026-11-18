# 任务13：阆苑84张到 D2/D3 诊断边界

- 更新：2026-11-18；状态：残差/尺度/边界审计完成，原生推进未开始；分支 `lane/task-13/langyuan-diagnostic`。
- 输入：阆苑女仙全量84张，scale=0.25；沿用任务5冻结 `base-2026-11-12`，不改生产代码、阈值、矩阵或正式基线。
- 依据：任务5输出 `/Users/dp/Desktop/raw-editor/output/acceptance/task-5/langyuan-baseline-r1` 及 [S2-langyuan](S2-langyuan.md)。

## 基线门禁

已有全量 0.25 运行走 `layered_virtual_tile`，28 个机位、84 个来源，closure 为 `converged`，峰值 RSS `15481438208` bytes（14.42 GiB），内存门通过；这只是 D1 诊断范围。

QG 九项中：`local_scale_pixel_ratio=fail`、`boundary_stroke_alignment=fail`；`effective_pixel_count`、`gradient_energy_normalized`、`noise_sigma_ratio`、`roi_delta_e00`、`owner_sharpness_coverage` 为 pass，MTF 与 `local_scale_median` 保持 not_applicable。非通过的原始位置：

- local-scale 8 个 ROI 中 6 个 pixel-ratio 为 `0.0`，对应 `DSC_3733`、`DSC_3711`、`DSC_3681`、`DSC_3731`、`DSC_3708`、`DSC_3727`；local median 最低为 `DSC_3681=0.6294107`；
- boundary 只有一个失败采样：world `(7457,1848.5)`，`station16↔27 / owner48↔82`，测值 `4px`；
- output-density protection 为 `bypassed`，原因是 `nonidentity_world_residual`，`sample_count=0`，不能用它证明 native 密度足够。

## 残差与尺度核查

closure 虽然收敛，但 `residual_median_px=0.0809119`、`residual_p95_px=1.28006397`，`max_direct_pair_p95_px=2.68102525`，`max_corner_correction_px=3.29073448`；residual warp 两个区域的 `p95_before_px` 为 `4.19886788` 与 `3.99492373`，均有 `reverted_cells=3`。因此 closure pass 不能抵消局部尺度与边界失败。

55 条已接受 station relations 的 `scale_ratio` 范围为 `0.8324044–1.2160185`，中位数 `1.0135168`；14 条低于 0.95。全量关系仍是一个 connectivity component，但 323 条关系被拒绝、302 条未测，主要原因/阶段为 `no_predicted_overlap=119`、`insufficient_bidirectional_matches=139`、`insufficient_measurable_patches=41`、`closure_unreliable_pair_p95=17` 和 `station_relation_scale_out_of_range=3`。

机器摘要：[langyuan-task-13-diagnostic.json](../benchmarks/langyuan-task-13-diagnostic.json)。

## 结论与下一步

本任务不跑 scale=1，也不把 10 张历史通过替代84张全量验收。下一轮应先做一个只读/test-only 的 station/residual 单变量实验：固定上述 world 坐标，记录 residual 前后、local scale、boundary、source relation 和 density-bypass 状态；在 0.25 重新验证后，才决定是否值得推进原生。参考图/导出后配准链不得反馈修改生产几何。

## residual 支持域审计

使用 [langyuan_residual_support_audit.py](../../scripts/stack-lab/langyuan_residual_support_audit.py) 对现有 Stack_Report 的失败 world 坐标与 residual field 支持区域做只读交集审计。报告中的两块 residual 区域约为 `x=6046–6564、y=1449–2199`；15 个失败记录（local median 8、local ratio 6、boundary 1）全部在区域外，交集为 `0`。

因此“强制 residual identity/禁用 residual”不会直接覆盖当前失败点；而且该开关还会解除 `nonidentity_world_residual` 的 density bypass，混入后续输出路径变化，不适合作为当前第一修复实验。机器结果保存在 `/Users/dp/Desktop/raw-editor/output/acceptance/task-13/langyuan-residual-support-audit.json`。下一步改做固定 station 的 source-to-world Jacobian sensitivity 审计，仍只读、不写生产 pose。

## station geometry sensitivity

对 14 个 local-scale failure records 做 source-to-world 只读 sensitivity：6 个 pixel-ratio=0 的点需要的理论 isotropic scale factor 为 `1.022799–1.509348` 才能达到 0.95；该计算只记录假设，不修改 homography、station pose 或 QG 阈值。机器结果保存在 `/Users/dp/Desktop/raw-editor/output/acceptance/task-13/langyuan-geometry-sensitivity-audit.json`。

这个跨度说明它不是一个可直接套用的全局倍率修复；下一步若实现，必须固定单一 station/单一 pose 假设并保存 before/after 的 local-scale、boundary、closure、density 和来源证据，先在 0.25 做完整 QG，再决定是否启动 scale=1。

## 几何 sensitivity 子集 A/B

对包含最坏点 `DSC_3681.NEF` 的 `langyuan-10` 子集做了 test-only pose sensitivity：绕虚拟 station group 中心施加 factor `1.5093482447`。hook 命中并记录 marker，但该高倍率导致 output-density factor `1.248047`、closure `no_constraints`，native acceptance 以 `native station geometry is not connected and converged` 拒绝；QG ROI 表面 pass 不能抵消这个 native 几何拒绝，self sharpness median 仅 `0.03345`。

因此停止把该倍率推广到84张全量。机器摘要：[langyuan-task-13-geometry-ab.json](../benchmarks/langyuan-task-13-geometry-ab.json)。
