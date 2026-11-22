# S2-lidong：李东任务17新素材

- 更新：2026-11-22；状态：**diagnostic_rejected_structural**（scale=0.25，独占，preview-only）。本卡记录 D1 与 bounded 分组诊断，不宣称 native/QG 通过。
- 输入目录：`/Users/dp/Downloads/书画拼图/李东`；扩展名：`.nef`；源图数：25；字节数：`1484471346`。
- 源集合 aggregate SHA-256：`17bedaf100ce2d06e001896b9dc373ad72b0cc80e8e5cbfbb54c4d0eddc76f85`；manifest：`tests/acceptance/focus-stack-datasets.json`；D1 中 25/25 解码，SHA 未变化，ownership verified=true。
- 冻结二进制：`base-2026-11-12`；binary SHA-256：`39bfe6f2aecc8a7ebc154efb94ba4b1f1a26257c825a676a0e54e8cde9cef03b`；未改生产算法、阈值或 bounds。
- 运行目录：`output/acceptance/task17/lidong-d1-r1/run`；结构化摘要：`output/acceptance/task17/lidong-d1-r1/analysis.json`。

## D1 结果

- 25 inputs（streamedSources=25）、11 stations；summary 输出 `2720×1942`；主路径 `layered_virtual_tile`。
- layered report：`stack-report-739353a6-0b5f-488b-9d3c-f1eea7d5268e.json`；degradation **rejected**，原因 `geometry_disconnected`，组件数 3；同时存在 station-relation scale-out-of-range / edge-strength mismatch、topology ambiguous、low-sharpness。tone、density、QG 均 **not_run**。
- legacy fallback report：`stack-report-b0d2d3a9-782f-4c08-a35d-99baf930ef34.json`；tone/density **not_run**；QG **fail**：`local_scale_pixel_ratio` 4/8 失败，`boundary_stroke_alignment` 5/74 失败，`owner_sharpness_coverage` 1/1；该 fallback 不能替代主路径验收。
- memory gate **PASS**：阈值 RSS `25,769,803,776`；layered 峰值 `11,428,921,344`，legacy 峰值 `11,490,295,808`；均未超限，线程 8。
- summary 的 `nativeAcceptance=passed-or-not-applicable` 仅是流程字段，不能解释为通过；结构拒绝已使 native acceptance 结论为 rejected，且没有 tone/density proof。

## Bounded grouping/overlap isolation

- 审计：[`task-17-lidong-grouping-overlap-bounded-r1.json`](../benchmarks/task-17-lidong-grouping-overlap-bounded-r1.json)；只读取既有 Stack_Report，未重跑 compositor。
- 三个断连组件映射为 C0={2,4}、C1={0,1,3,5,6,8,9}、C2={7,10}；accepted cross-component edges 为 0。
- 最强 C0↔C1 候选 4↔5：111 bidirectional/inliers、support 0.654、holdout 0.783/0.649、scale 1.2689；最强 C1↔C2 候选 6↔10：130 inliers、support 0.679、holdout 1.809/0.637、scale 1.3642；均因现有 `station_relation_scale_out_of_range` 被拒。
- synthetic graph A/B 仅显示强行注入一条边得到 2 components、两条边得到 1 component，但两者都与现有 gate 不兼容；不能据此合并真实组件。
- bounded 分组/重叠实验：[`task-17-lidong-grouping-overlap-bounded-r1.json`](../benchmarks/task-17-lidong-grouping-overlap-bounded-r1.json)。三组件间没有 gate-compatible bridge；最强 C0↔C1 与 C1↔C2 候选均因 `station_relation_scale_out_of_range` 被拒。合成强连边只能改变组件数，不能形成可接受关系。

## 结论与边界

李东 D1 是结构性诊断拒绝：source contract 和内存门通过，但几何连通性失败，fallback 也未通过质量门。bounded 分组/重叠审计没有找到 gate-compatible bridge，因此不进入 D2 native、D3 QG、D4 人审或发布验收；若继续，只能构造明确的 source regrouping/真实重叠 fixture，不能放宽关系门或强行连边。
