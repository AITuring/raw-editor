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

当前仍是 D1 的 scale=0.25 诊断范围，任务12不勾选 D3。最小 fallback 假设没有收益证据：尺度失败点没有可替代 owner，owner 覆盖失败点没有 Coverage 支持。下一次若进入实现，必须先做独立 source resampling 并重新计算 local-scale、owner coverage、boundary、detail donor 和完整 QG；不得自动替换 owner、放宽阈值或把 0.25 结果外推到 native。
