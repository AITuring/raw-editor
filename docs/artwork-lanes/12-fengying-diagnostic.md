# 任务12：枫鹰到 D3 诊断边界

- 更新：2026-11-18；状态：诊断完成，修复未开始；分支 `lane/task-12/fengying-diagnostic`；起点 v1 `43f5668e`。
- 输入范围：枫鹰三文件夹全量 110 张，scale=0.25；使用同一冻结 `base-2026-11-12`，不改生产算法、质量阈值或正式基线。
- 目标：按 S2 卡定位 DSC04916 的局部尺度失败与 owner 覆盖失败，并补齐 test-only owner/coverage 导出，判断 3×3 fallback 是否有足够证据。

## 重跑与门禁

运行目录：`/Users/dp/Desktop/raw-editor/output/acceptance/task-12/fengying-diagnostics-r1`。

```sh
python3 scripts/stack-lab/run.py \
  --binary base-2026-11-12 --lane task-12 --dataset fengying \
  --scale 0.25 --label diagnostics-r1 --exclusive --streaming-diagnostics
```

运行使用 exclusive 槽、Rayon=8；冻结二进制 SHA 为
`39bfe6f2aecc8a7ebc154efb94ba4b1f1a26257c825a676a0e54e8cde9cef03b`，原始构建提交为
`c0c1e5a9037fb4f8a8f8b983802848feb667b9ba`。原生子进程退出码为 0，runner 因 QG rejected 返回 1；峰值 RSS `13988659200` bytes（13.02 GiB），24 GiB 内存门通过，wall `1934.73s`。

QG 与 S2 基线一致：`legacy_single_layer_mosaic`、closure `not_run`、104 个报告来源、6 张渲染 JPEG 仍按 manifest 排除、unresolved=0；非通过项为 `local_scale_pixel_ratio`、`owner_sharpness_coverage`，其余边界/噪声/颜色门保持原结果。预览及六区裁片 digest 与任务5基线一致，未把 TIFF 字节差异解释成质量变化。

## owner/coverage 诊断

新增 `scripts/stack-lab/run.py --streaming-diagnostics`，只设置
`RAW_EDITOR_STREAMING_MOSAIC_DIAGNOSTICS_DIR`，与 `--framed` 保护模式分离；执行记录会保存该开关及导出目录。新增审计脚本
[fengying_owner_fallback_audit.py](../../scripts/stack-lab/fengying_owner_fallback_audit.py)，只读诊断图，不写像素、不改变 owner。

导出目录为 `/Users/dp/Desktop/raw-editor/output/acceptance/task-12/fengying-diagnostics-r1/diagnostics`，analysis 平面为 `2048×1716`、scale `4.82080078125`，source union crop `[35,12,9873,8271]`。以 S2 的 world origin `(-4068,-5462)` 映射失败点并检查同一 3×3 stencil：

- 9 个 `local_scale_median` 失败采样（其中 DSC04916 同时命中 `local_scale_pixel_ratio`）全部是同一 owner 的 9/9 邻域，0 个 alternate owner candidate；
- owner 覆盖失败点的 3×3 邻域为 owner=0 且 Coverage=0，0 个有覆盖的邻居；
- 本次没有 boundary 失败点，因此不把 boundary pass 当作原生完成证据。

审计机器摘要：[fengying-task-12-diagnostic.json](../benchmarks/fengying-task-12-diagnostic.json)；完整 fallback 结果在 `/Users/dp/Desktop/raw-editor/output/acceptance/task-12/fengying-owner-fallback-audit.json`。

## 结论与下一步

当前仍是 D1 的 scale=0.25 诊断范围，任务12不勾选 D3。最小 fallback 假设没有收益证据：尺度失败点没有可替代 owner，owner 覆盖失败点没有 Coverage 支持。不得自动替换 owner、放宽阈值或把 0.25 结果外推到 native；后续应转向几何倍率与 owner coverage 根因。

## 独立 source resampling A/B

在同一输入范围上建立 test-only 实验二进制 `task-12-source-resample-r1`，只对
`DSC04916.ARW` 强制 source divisor `1.25`，其余 103 个 RAW 保持默认预滤波；实验通过
`cfg(test)` 开关读取 `RAW_EDITOR_TEST_SOURCE_RESAMPLE_PATH` 和
`RAW_EDITOR_TEST_SOURCE_RESAMPLE_DIVISOR`，并把 divisor 写入 native registration，使 QG
reference 重放同一 source sampling。运行目录为
`/Users/dp/Desktop/raw-editor/output/acceptance/task-12/fengying-resample-dsc04916-r1`。

完整 exclusive 重跑的峰值 RSS 为 `14183464960` bytes（13.21 GiB），native child exit=0，内存门通过。QG 九项逐项重算后仍为：

- `local_scale_pixel_ratio=fail`，DSC04916 仍为 `0.0`；几何 local scale 仍为 `0.8887075`；
- `owner_sharpness_coverage=fail`，`0.9713647986 → 0.9713504992`，覆盖/拥有像素减少 38；
- `effective_pixel_count`、`gradient_energy_normalized`、`noise_sigma_ratio`、`roi_delta_e00`、`boundary_stroke_alignment` 保持 pass，MTF 保持 not_applicable，unresolved=0；
- 预览与六个裁片 digest 全部改变，self sharpness median `0.16457 → 0.16594`，但没有门禁收益；detail repairs `409 → 405`。

QG source visual capture 完整覆盖失败 ROI：`roi-02` 对应 station 2 / owner 59 / `DSC04916.ARW`，world origin `(-3428,-4822)`，`65536/65536` 像素有效映射，精确 RGB32F output/reference、prepared source XY、owners 和 validity 均已保存。机器摘要见
[fengying-task-12-resampling.json](../benchmarks/fengying-task-12-resampling.json)。

结论：该单源 resampling 只改变像素与锐度，不能改变几何 Jacobian 或制造 owner Coverage；不进入生产 resampling/fallback。下一步应转向 source-to-world 几何倍率和 owner coverage 根因。

## source-to-world geometry sensitivity（只读）

对保存的 9 个 local-scale failure records 做 source-to-world sensitivity：只有 DSC04916.ARW 的测值低于 0.95，理论 isotropic factor 为 `1.068968`；其余失败记录测值在 `0.99968–1.41569`，不支持统一倍率修复。审计只保存 world 坐标、source id、原始 homography 和理论 factor，不修改 pose、owner、阈值或像素。

机器摘要：[fengying-task-12-geometry-sensitivity-audit.json](../benchmarks/fengying-task-12-geometry-sensitivity-audit.json)。这个结果支持继续定位 DSC04916 的局部 source-to-world/owner 根因，但不授权把 factor 直接写入生产路径。

## DSC04916.ARW 单源 geometry A/B（已完成）

使用 test-only legacy source-scan pose hook，对 source id=58 的 `DSC04916.ARW` 施加 sensitivity audit 得到的 factor `1.06896813096`，完成三文件夹 110 输入、scale=0.25 独占 QG 验证。

- `local_scale_pixel_ratio`：`fail (0.0) → pass`；
- `owner_sharpness_coverage`：`0.9713647986 → 0.9708641435`，仍 fail；
- `boundary_stroke_alignment`：原本 pass，新增 2 个 `2px` 失败；
- detail repairs：`409 → 406`；24GiB 内存门通过；来源计数保持一致；最终 QG 仍 fail。

结论：该倍率只能修复单一尺度门，代价是 owner/边界退化，不能进入生产默认，也不能授予 D2/D3。机器摘要：[fengying-task-12-legacy-geometry-ab.json](../benchmarks/fengying-task-12-legacy-geometry-ab.json)。实验 hook 已从主线移除。
