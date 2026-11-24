# 任务13：阆苑84张到 D2/D3 诊断边界

- 更新：2026-11-23；状态：残差/尺度/boundary 根因诊断收口；scale=1 独占资源试跑完成但 D3 未通过，保留拒绝语义；分支 `lane/task-13/langyuan-diagnostic`。
- 任务12交接（2026-11-22）：枫鹰 fallback、resampling 与 geometry factor 诊断已收口且无生产合入；本卡继续沿用 exclusive 资源和不把 scale=0.25 结果外推到 native 的边界。
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

诊断阶段曾不把 scale=0.25 外推到 native，也不把10张历史通过替代84张全量验收。后续的 scale=1 资源试跑已按独占槽完成；参考图/导出后配准链不得反馈修改生产几何。

## residual 支持域审计

使用 [langyuan_residual_support_audit.py](../../scripts/stack-lab/langyuan_residual_support_audit.py) 对现有 Stack_Report 的失败 world 坐标与 residual field 支持区域做只读交集审计。报告中的两块 residual 区域约为 `x=6046–6564、y=1449–2199`；15 个失败记录（local median 8、local ratio 6、boundary 1）全部在区域外，交集为 `0`。

因此“强制 residual identity/禁用 residual”不会直接覆盖当前失败点；而且该开关还会解除 `nonidentity_world_residual` 的 density bypass，混入后续输出路径变化，不适合作为当前第一修复实验。机器结果保存在 `/Users/dp/Desktop/raw-editor/output/acceptance/task-13/langyuan-residual-support-audit.json`。下一步改做固定 station 的 source-to-world Jacobian sensitivity 审计，仍只读、不写生产 pose。

## station geometry sensitivity

对 14 个 local-scale failure records 做 source-to-world 只读 sensitivity：6 个 pixel-ratio=0 的点需要的理论 isotropic scale factor 为 `1.022799–1.509348` 才能达到 0.95；该计算只记录假设，不修改 homography、station pose 或 QG 阈值。机器结果保存在 `/Users/dp/Desktop/raw-editor/output/acceptance/task-13/langyuan-geometry-sensitivity-audit.json`。

这个跨度说明它不是一个可直接套用的全局倍率修复；下一步若实现，必须固定单一 station/单一 pose 假设并保存 before/after 的 local-scale、boundary、closure、density 和来源证据，先在 0.25 做完整 QG，再决定是否启动 scale=1。

## 几何 sensitivity 子集 A/B

对包含最坏点 `DSC_3681.NEF` 的 `langyuan-10` 子集做了 test-only pose sensitivity：绕虚拟 station group 中心施加 factor `1.5093482447`。hook 命中并记录 marker，但该高倍率导致 output-density factor `1.248047`、closure `no_constraints`，native acceptance 以 `native station geometry is not connected and converged` 拒绝；QG ROI 表面 pass 不能抵消这个 native 几何拒绝，self sharpness median 仅 `0.03345`。

因此停止把该倍率推广到84张全量。机器摘要：[langyuan-task-13-geometry-ab.json](../benchmarks/langyuan-task-13-geometry-ab.json)。

## 根因诊断收口与任务交接

full-84 test-only control 重跑复现了同一 QG：6/8 local-scale pixel-ratio 失败、唯一 boundary witness 为 station16↔27 / owner48↔82 的 4px，owner coverage 通过；closure 仍 converged，residual median/P95 `0.0809/1.2801px`，density 仍因 `nonidentity_world_residual` bypass。该重跑的 Stack_Report 已写出，但参考注册失败导致 acceptance metrics 未写出，因此不把它当作新的验收记录；QG 与 baseline 报告一致。

综合 residual support、station scale sensitivity、boundary witness 和既有 station5 高倍率子集证据，任务13诊断结论是：当前失败点不在记录的 residual 支持域内；尺度因子分散于 `1.0228–1.5093`，不支持全局倍率；boundary 是独立的 station16↔27 覆盖描线问题。停止继续扫描 factor、禁用 residual、放宽阈值、bounds 或 fallback；scale=1 仅作为资源可行性试跑，不能转化为生产验收。

机器摘要：[`langyuan-task-13-root-cause-closure-r1.json`](../benchmarks/langyuan-task-13-root-cause-closure-r1.json)。下一项交接任务14文苑，沿用不把 scale=0.25 诊断外推到 native 的约束。

## scale=1独占资源试跑

2026-11-23 使用已验证的 `task-13-focus-scale-r1` 冻结二进制（commit `1b638eb278f2749795b87a8780a86f91b757086b`，binary SHA `20a54e35e02586dd030b4cef0c8131417da24c7dbebae539d3d75c9ba4dc4b27`）对阆苑84张、scale=1执行一次 `exclusive` 运行。84/84输入预检通过，没有空源或物理缺失源；资源合同为 preparation/Rayon `1/1`，内存门阈值24GiB。

运行完成但被原生验收拒绝：wall `9829.78s`，峰值 RSS `13,631,864,832B`（12.69GiB），内存门通过；TIFF已写出且RGBA16/sRGB/alpha和artifact validation通过，尺寸 `34670×10560`，输出 SHA `6c4e297bfa507c8d82909d2a0ab53a622af15b8439d7c9e2820e78a7272787c2`，运行目录约 `2.9GiB`。但 harness 报告 verified TIFF 尺寸与渲染结果不一致，整体输出合同不接受；原生 harness exit `101`，没有 acceptance metrics；closure 为 `unreliable`，max direct-pair p95 `3.5276637618px`，fallback `closure_unreliable_pair_p95`；output density 因 `invalid_native_jacobian_probes`（`sample_count=0`、`invalid_probe_count=82`）bypass。QG fail：`local_scale_median`、`local_scale_pixel_ratio` fail，`effective_pixel_count`、gradient、noise、ΔE、owner coverage pass；统一 `analysis.json` 状态为 `rejected`，不授予D2/D3。

证据：[机器摘要](../benchmarks/task-4c-langyuan-scale1-pilot-r1.json)、[execution.json](/Users/dp/Desktop/raw-editor/output/acceptance/task-4c/langyuan-scale1-r1/execution.json)、[analysis.json](/Users/dp/Desktop/raw-editor/output/acceptance/task-4c/langyuan-scale1-r1/analysis.json)、[Stack_Report](/Users/dp/Desktop/raw-editor/output/acceptance/task-4c/langyuan-scale1-r1/run/langyuan-s1-task-4c-scale1-r1-stack-reports/stack-report-c8bf6f03-56f1-4603-a5b6-bf55b1173c5c.json)。本次结果只完成资源可行性测量，不放宽门禁、不修改生产阈值/bounds/fallback，不把TIFF写出当作验收通过。

### 试跑后的只读复核

使用现有 `langyuan_residual_support_audit.py` 对本次 `analysis.json` 与 `Stack_Report` 做交集审计：10 个 local-scale 失败点全部在记录的 residual 支持区域之外（inside `0`、outside `10`；本报告 `residual_warp.regions=[]`、`identity=true`）。这排除了“禁用 residual/强制 residual identity”作为当前失败解释。只读汇总见 [task-4c follow-up audit](../benchmarks/task-4c-langyuan-scale1-followup-audit-r1.json)。下一步只保留 station/source Jacobian 与 closure offending-pair witness 的 test-only 审计，完成前不重复 native 全量运行。

source-to-world sensitivity 随后完成：4 个非零 local-scale failure 点达到 `0.95` 所需的理论各向同性因子为 `1.08615–1.76232`；0 值 pixel-ratio 点不能反推倍率，跨度不支持全局倍率修复。结果仅作为几何假设证据，不写入 pose、bounds、阈值或 fallback：[scale1 geometry sensitivity](../benchmarks/task-4c-langyuan-scale1-geometry-sensitivity-r1.json)。

closure offender witness 审计也已完成：station `0↔17` 的候选为 fitted projective relation，`174` 个 fitted inliers、selected scale ratio `1.17565`，但 spatial support 仅 `0.42051`；其 direct closure P95 仍为 `3.52766px`，超过 `3px` 门。该候选不是生产 pose 修正证据；后续只审计这一对的 correspondence support 与 source/Jacobian 关系，不放宽 closure 门禁：[closure offender audit](../benchmarks/task-4c-langyuan-scale1-closure-offender-r1.json)。

与同一报告的 27 条 accepted relation 对照后，offender 的 spatial support `0.42051` 和 median error `1.26287px` 落在 accepted 分布内，但 fitted inliers `174` 低于 accepted 最小值 `249`，scale ratio `1.17565` 高于 accepted 最大值 `1.11199`。这把后续问题收敛到低支持量与异常尺度关系的 source/Jacobian 证据，不授权扩大 scale bounds：[support comparison](../benchmarks/task-4c-langyuan-scale1-closure-support-comparison-r1.json)。

进一步复核两个 station 的 `tile_to_world` Jacobian：station 0 的 `sqrt(|det|)=1.0000`，station 17 为 `0.84775`，其倒数 `1.17960` 与拟合 relation scale `1.17565` 接近；两站都标为 `topology_index_ambiguous`。这说明尺度差异具有稳定几何证据，不能按随机 correspondence 漂移处理；当前仍受低 inlier 支持和 topology ambiguity 阻断：[station Jacobian audit](../benchmarks/task-4c-langyuan-scale1-station-jacobian-r1.json)。

至此本轮 bounded correspondence-support 与 source/Jacobian witness 审计闭合，未产生 pose、bounds、阈值或 fallback 修改；除非有新的拓扑或来源证据，否则不再重跑该 84 张全量 native。

随后完成 topology/source correspondence 只读审计：station 0 与 17 的成员集合均与 `source-alignment.json` 的 capture group 精确一致，排除了漏源或错分组解释；但 station 17 的组内 center drift median/max 为 `139.21/211.43px`，station 0 为 `50.25/55.54px`（约 `2.77x/3.81x`），两站仍标记 `topology_index_ambiguous`。这进一步支持“拓扑/物理倍率诊断”而非全局 scale 修正：[topology correspondence audit](../benchmarks/task-4c-langyuan-scale1-topology-correspondence-r1.json)。

本轮统一收口摘要：[scale=1 closure](../benchmarks/task-4c-langyuan-scale1-closure-r1.json)。
