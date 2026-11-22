# S2-chibi：赤壁任务17新素材

- 更新：2026-11-22；状态：**diagnostic_rejected / quality_gate_failed**（scale=0.25，独占，preview-only）。本卡记录 D1 与 bounded isolation 证据，不宣称 native/QG 通过。
- 输入目录：`/Users/dp/Downloads/书画拼图/赤壁`；扩展名：`.nef`；源图数：48；字节数：`2725524205`。
- 源集合 aggregate SHA-256：`d41a9207807ea6bc6bb60d2d71491dbc2e02f77bb059746cd6104bd6ed20b0b8`；manifest：`tests/acceptance/focus-stack-datasets.json`；D1 中 48/48 解码，SHA 未变化。
- 冻结二进制：`base-2026-11-12`；binary SHA-256：`39bfe6f2aecc8a7ebc154efb94ba4b1f1a26257c825a676a0e54e8cde9cef03b`；未改生产算法、阈值或 bounds。
- 运行目录：`output/acceptance/task17/chibi-d1-r1/run`；Stack_Report：`output/acceptance/task17/chibi-d1-r1/run/chibi-s0p25-task17-chibid1r1-stack-reports/stack-report-504a289a-2351-43bc-a4b8-0b89d25a3110.json`；结构化摘要：`output/acceptance/task17/chibi-d1-r1/analysis.json`。
- 资源合同：独占 run 槽；`RAYON_NUM_THREADS=8`；memory gate **PASS**，阈值 RSS `25,769,803,776` bytes，峰值 RSS `12,401,836,032` bytes，线程数 8。

## D1 结果

- 运行目录：`output/acceptance/task17/chibi-d1-r1/run`；48 inputs，18 stations；输出尺寸 `7637×2155`。
- tone：`degraded`。boundary 最大 ΔE00=`10.0428838189`，位置 `(x=3584,y=256)`，pair `2↔5`；共 13 个 tone boundary violations。
- QG：**fail**。`local_scale_pixel_ratio=6/8`、`boundary_stroke_alignment=4/155`；effective pixel、gradient、ROI ΔE00、owner sharpness 项通过。
- source contract：48/48 解码，aggregate SHA 未变化。
- output density：`bypassed`；sampling contract 为 `sampled_native_jacobian_not_full_pixel_proof`。
- 后续 bounded isolation：[`task-17-chibi-tone-local-scale-isolation-r1.json`](../benchmarks/task-17-chibi-tone-local-scale-isolation-r1.json)。2↔5 candidate 仅 10 个 measurable patches/8 个 bidirectional matches，`residual_model=not_run`、`failure_stage=insufficient_measurable_patches`、prefit orientation difference≈14.043°，且未进入 tone native fitting relations；6 个 local-scale pixel-ratio failures 映射到 stations 2/3/5/7/8/11。该审计未改阈值、bounds 或 fallback。
- bounded path-only source witness：[`task-17-chibi-tone-source-mapping-audit-r3.json`](../benchmarks/task-17-chibi-tone-source-mapping-audit-r3.json)；station2/5 的四个 source paths 均 loaded，8 个 probes 均 `native_registration=true`、`native_warp=true`，但 native source sampling 全部 `native_source_sampling_unmapped`，在 RGB sampling 前就阻断，因此没有形成 photometric relation 结论。r1/r2 的历史 loader 记录保留；derivative tone/QG 仅作 mapping-blocked diagnostic evidence，不覆盖原始 D1。

## 结论与边界

该结果是 D1 诊断拒绝：内存门通过，但 tone/boundary 与局部尺度质量门失败，因此不进入 D2 native、D3 QG、D4 人审或发布验收。2↔5 source witness 当前被 native source sampling footprint/ownership mapping 阻断，不能把失败解释成 photometric mismatch。保持生产阈值、tone bounds、local-scale threshold 与 fallback 不变；任务17赤壁线在此收口。
