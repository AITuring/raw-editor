# S2-xiagui：夏圭任务17新素材

- 更新：2026-11-22；状态：**diagnostic_rejected / quality_gate_failed**（scale=0.25，独占，preview-only）。本卡记录 D1 与 bounded capture 证据，不宣称 native/QG 通过。
- 输入目录：`/Users/dp/Downloads/书画拼图/夏圭`；扩展名：`.nef`；源图数：48；字节数：`2903839002`。
- 源集合 aggregate SHA-256：`338af8d29e488d8c4d772677ea3471ece805c3c7fa7ec5b5b1ec6f5a3672fc06`；manifest：`tests/acceptance/focus-stack-datasets.json`；D1 中 48/48 解码，SHA 未变化。
- 冻结二进制：`base-2026-11-12`；binary SHA-256：`39bfe6f2aecc8a7ebc154efb94ba4b1f1a26257c825a676a0e54e8cde9cef03b`；未改生产算法、阈值或 bounds。
- 运行目录：`output/acceptance/task17/xiagui-d1-r1/run`；Stack_Report：`output/acceptance/task17/xiagui-d1-r1/run/xiagui-s0p25-task17-xiaguid1r1-stack-reports/stack-report-cb3753ca-4a6b-45ae-9044-99fa72700b35.json`；结构化摘要：`output/acceptance/task17/xiagui-d1-r1/analysis.json`。

## D1 结果

- 48 inputs、15 stations；summary 输出 `3395×2922`；selected path `layered_virtual_tile`；tone **Applied**（15 tiles），boundary 最大 ΔE00=`1.3806623592`（阈值 1.5，无 tone boundary violation）。
- QG **fail**：`boundary_stroke_alignment` 7/46 失败，失败值约 `2.4287–3.5658 > 1.5`；`local_scale_median` 为 diagnostic N/A（8/8），`local_scale_pixel_ratio`、effective pixel、gradient、ROI ΔE00、owner sharpness 通过。
- station relation：拒绝 13 条，其中 11 条为 `closure_unreliable_pair_p95`。
- output density：`applied`，factor `1.201171875`；samples `127207`，invalid `0`，boundary probes `4542`；direct p95 `2.35147`，residual median `0.342725`，max corner `5.14225`；minimum/p01/median scale `0.83284 / 0.863529 / 0.948205`；sampling contract=`sampled_native_jacobian_not_full_pixel_proof`。
- memory gate **PASS**：阈值 RSS `25,769,803,776`，峰值 RSS `12,656,295,936`，未超限；线程 8，最大 virtual tiles 16。

## Boundary witness replay

- 审计：[`task-17-boundary-witness-replay-r1.json`](../benchmarks/task-17-boundary-witness-replay-r1.json)；native witnesses `(-752,1)`→preview `(30,0)`、`(-439,2304)`→preview `(140,813)`，native measured `2.4287–3.5658`。
- 两个 saved-preview replay 的 fits/edge 均返回 `boundary_low_contrast`；cargo test 通过，但这是内容不适用的缩略图探针，不能推翻 native QG 失败，也没有提供 owner/source 证据。
- raw-owner capture（pre-fix debug region）：[`task-17-xiagui-native-boundary-capture-r1.json`](../benchmarks/task-17-xiagui-native-boundary-capture-r1.json)；同冻结 binary/scale=0.25 加入 test-only capture sinks 后，7 个 native failures 均生成 96px failed-boundary crop。`(-752,1)` 落在输出 `(0,0)`、owner=0，固定 debug 窗口为空；`(-439,2304)` 一类落在底部边界附近。候选文件未 flush，不能把这次 capture 当作修复或验收。
- 动态 debug/P95 derivative capture：[`task-17-xiagui-native-boundary-capture-r3.json`](../benchmarks/task-17-xiagui-native-boundary-capture-r3.json)；主 witness 移到真实 covered output `(533,1637)`，邻域 `625/625` covered、owner zero `0`，主要 owner `11→DSC_0988.NEF`（station3）与 `27→DSC_0980.NEF`（station9），3↔9 几何 relation 通过（174 inliers、support `0.5247`、scale `1.05027`）。boundary-stroke 仍 fail；该 derivative 同时出现 tone ΔE00 max `4.116856`/5 violations，保留为 test-only evidence，不与原始 D1 tone 结果混合，也不推广到生产 bounds。

## 结论与边界

夏圭 D1 是 native acceptance rejected 的诊断结果：source contract 完整、原始 D1 tone 边界值在阈值内、内存通过，但 boundary-stroke alignment 质量门失败。动态 derivative 已证明主 witness 不是 origin/owner-zero artifact，而是 covered 的 station3↔9 boundary evidence；derivative tone ΔE00 越界仅作为诊断分支保留，不能覆盖原始 D1，也不授权生产 tone 修正。preview witness replay 只能说明缩略图处无可配对边缘，因此不进入 D2 native、D3 QG、D4 人审或发布验收；保持生产阈值、bounds 与 fallback 不变。
