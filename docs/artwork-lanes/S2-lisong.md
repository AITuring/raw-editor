# S2-lisong：李嵩任务17新素材

- 更新：2026-11-22；状态：**diagnostic_rejected / quality_gate_failed**（scale=0.25，独占，preview-only）。本卡记录 D1 与 bounded capture 证据，不宣称 native/QG 通过。
- 输入目录：`/Users/dp/Downloads/书画拼图/李嵩`；扩展名：`.nef`；源图数：46；字节数：`2818722255`。
- 源集合 aggregate SHA-256：`760a30788bd2555c260d3b246217ddaf098b3a0ab50cb826297d336b0d639007`；manifest：`tests/acceptance/focus-stack-datasets.json`；D1 中 46/46 解码，SHA 未变化。
- 冻结二进制：`base-2026-11-12`；binary SHA-256：`39bfe6f2aecc8a7ebc154efb94ba4b1f1a26257c825a676a0e54e8cde9cef03b`；未改生产算法、阈值或 bounds。
- 运行目录：`output/acceptance/task17/lisong-d1-r1/run`；Stack_Report：`output/acceptance/task17/lisong-d1-r1/run/lisong-s0p25-task17-lisongd1r1-stack-reports/stack-report-6b797c3f-2301-4742-85c6-808b6e145327.json`；结构化摘要：`output/acceptance/task17/lisong-d1-r1/analysis.json`。

## D1 结果

- 46 inputs、15 stations；summary 输出 `3205×2194`；selected path `layered_virtual_tile`；tone **Applied**，boundary 最大 ΔE00=`0.5645683266`（阈值 1.5，无 tone boundary violation）。
- QG **fail**：仅 `boundary_stroke_alignment` 失败，1/6，失败值 `1.6581128181810776 > 1.5`；`local_scale_median` 为 diagnostic N/A（8/8，范围 `1.0503176276–1.1845703125`），`local_scale_pixel_ratio`、effective pixel、gradient、ROI ΔE00、owner sharpness 通过；noise 为 N/A。
- station relation：拒绝 1 条，原因 `closure_unreliable_pair_p95`。
- output density：`applied`，factor `1.1845703125`；samples `121955`，invalid `0`，boundary probes `3446`；direct p95 `2.4716521291`，residual median `0.2345679095`；minimum/p01/median scale `0.8445761861 / 0.8689395382 / 0.9437522348`；sampling contract=`sampled_native_jacobian_not_full_pixel_proof`。
- memory gate **PASS**：阈值 RSS `25,769,803,776`，峰值 RSS `12,670,844,928`，未超限；线程 8，最大 virtual tiles 16。

## Boundary witness replay

- 审计：[`task-17-boundary-witness-replay-r1.json`](../benchmarks/task-17-boundary-witness-replay-r1.json)；native witness `world=(-741,-5)`、native measured `1.6581` 映射到 1200px preview `(18,2)`。
- saved-preview replay 的 fits/edge 均返回 `boundary_low_contrast`；cargo test 通过，但这是内容不适用的缩略图探针，不能推翻 native QG 失败，也没有提供 owner/source 证据。
- raw-owner capture：[`task-17-lisong-native-boundary-capture-r1.json`](../benchmarks/task-17-lisong-native-boundary-capture-r1.json)；同冻结 binary/scale=0.25 加入 test-only capture sinks 后，1 个 native failure 生成 96px crop。witness 落在输出 `(0,0)`、owner=0，固定 debug 窗口为空；候选文件未 flush，不能把这次 capture 当作修复或验收。
- 动态 debug region 验证：[`task-17-lisong-native-boundary-capture-r2.json`](../benchmarks/task-17-lisong-native-boundary-capture-r2.json)；诊断派生 binary SHA=`b089029227122dba3146b9a6d078e35849d503d05e822608138bcfc64724affd`，同一 witness 生成有效 `500×500` raw-owner map、18 个 unique owners，anchor=`(0,0)`/owner=0；native boundary failure 仍存在（测得 `1.7252`），因此只是修复诊断出口，不是质量修复。
- edge attribution：[`task-17-edge-owner-coverage-audit-r1.json`](../benchmarks/task-17-edge-owner-coverage-audit-r1.json)；witness neighborhood `169/169` 像素均为 `owner=0`、`coverage=0`、`luma=0`，分类为 `edge_uncovered_owner_zero`。这解释了 source provenance 为空，但保留 native QG failure，不把它改成 pass。
- origin aggregation：[`task-17-boundary-origin-aggregation-audit-r1.json`](../benchmarks/task-17-boundary-origin-aggregation-audit-r1.json)；native failure 正好记录在 cropped world origin，说明当前 lead 是 boundary criterion/crop attribution 语义，而不是 tone bounds 或 source tone mismatch。
- p95 attribution 修正：[`task-17-boundary-p95-coordinate-fix-r1.json`](../benchmarks/task-17-boundary-p95-coordinate-fix-r1.json)；未来报告将 aggregate P95 指向真实 measured boundary point，P95 数值和 QG 阈值保持不变；历史 D1 报告保留为 pre-fix evidence。
- 修正后 replay：[`task-17-lisong-native-boundary-capture-r3.json`](../benchmarks/task-17-lisong-native-boundary-capture-r3.json)；witness 已从未覆盖 origin 移到真实 boundary output `(3010,1697)`，动态 owner map 有效且覆盖非零；native QG 仍 fail（`1.7252`），现在属于可追踪的真实 boundary evidence，而不是 origin attribution artifact。
- source-owner fixture：[`task-17-lisong-boundary-source-owner-fixture-r1.json`](../benchmarks/task-17-lisong-boundary-source-owner-fixture-r1.json)；witness 由 owners `30/41/44`（`DSC_0910` station 9 与 `DSC_0905/0912` station 13）组成，全部 coverage 有效；9↔13 station relation 已通过几何门（275 inliers、support 0.659、scale 1.0016、median error 0.289），但 boundary stroke 仍为 `1.7252`，归类为真实 source-owner/stroke mismatch。
- bounded source RGB sampler：[`task-17-lisong-source-rgb-sampler-r1.json`](../benchmarks/task-17-lisong-source-rgb-sampler-r1.json) 与 [`task-17-lisong-source-rgb-sampler-r2.json`](../benchmarks/task-17-lisong-source-rgb-sampler-r2.json) 保留为历史 partial evidence；path-only 强制 availability 与三源采样见 [`task-17-lisong-source-rgb-sampler-r3.json`](../benchmarks/task-17-lisong-source-rgb-sampler-r3.json)，saved-tone relation/content 审计见 [`task-17-lisong-saved-tone-relation-content-audit-r1.json`](../benchmarks/task-17-lisong-saved-tone-relation-content-audit-r1.json)，3×3 path-only ring 复核见 [`task-17-lisong-source-rgb-ring-audit-r1.json`](../benchmarks/task-17-lisong-source-rgb-ring-audit-r1.json)。ring 的同 station 13 luma 中位数为 `1.00113`，范围 `0.96694–1.13877`；中心差异没有在邻域内形成稳定 uniform gain，归类为 localized source-content/boundary sensitivity，仍不直接导出生产修正。tone boundary ΔE00 最大 `0.919580 < 1.5`，density `applied`（factor `1.1845703125`），memory gate 通过；native boundary-stroke 仍以 `1.725157 > 1.5` 拒绝。

## 结论与边界

李嵩 D1 是 native acceptance rejected 的诊断结果：source contract 完整、tone 数值在边界内、density applied、内存通过，但 boundary-stroke alignment 质量门失败。path-only source RGB fixture 已补齐三源实际采样；saved-tone relation/content audit 与 3×3 ring 复核显示几何 relation 通过、saved tone 近 1，但中心差异在邻域内不稳定，更符合 localized source-content/boundary sensitivity，而不是可推广的 uniform tone gain。该证据不能授权 tone model 修改。preview witness replay 只能说明缩略图处无可配对边缘，因此不进入 D2 native、D3 QG、D4 人审或发布验收；保持生产阈值、tone bounds 与 fallback 不变。
