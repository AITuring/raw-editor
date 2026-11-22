# 任务11：祁序到 D3 诊断边界

- 更新：2026-11-22；状态：诊断收口，D3 未通过；停止 factor 扫描，交接任务12；分支 `lane/task-11/qixu-diagnostic`；起点 v1 `4d7d33aa`。
- 输入范围：祁序1 75 张、祁序2 98 张，均为任务5冻结 baseline、scale=0.25；本次不改生产代码、阈值、基线或输出。

## 基线失败摘要

| 数据集 | 主合成/报告来源 | QG非通过 | 关键值 | 输出峰值 |
| --- | --- | --- | --- | ---: |
| 祁序1 | 70 streamed、72 report sources；4 张排除/缺少排序记录 | local scale ratio、boundary、owner coverage | 6 个 ROI scale=0；3 个 boundary=3px；owner coverage `0.9113826565` | 14.08 GiB |
| 祁序2 | 96 streamed、97 report sources；2 张排除/缺少排序记录 | local scale ratio、noise ratio、boundary、owner coverage | `DSC_6796` noise ratio `1.2391777416`；boundary=3px；owner coverage `0.8734524134` | 13.86 GiB |

两套都走 `legacy_single_layer_mosaic`，closure 为 `not_run`，未解析像素为 0，但这不抵消几何/owner/QG 失败。两套还存在输入排序与最终 Stack_Report 来源数不一致：祁序1 的 `DSC_3842/3910/3911` 无法对齐、`DSC_3879` 被软区筛除；祁序2 的 `DSC_6708` 无法对齐、`DSC_6730` 被软区筛除。正式 runner 因这些来源合同差异拒绝，诊断报告不能替代正式输入合同。

## 失败位置与含义

- 祁序1 local-scale 失败点集中在 `DSC_3850/3855/3860/3863/3867/3882` 的 owner 路径，另有 3 个跨 station boundary 采样达到 3px；这更像稀疏/错位的跨机位几何与 owner 覆盖问题，不是单一色调问题。
- 祁序2 的 local-scale 失败点分散在 `DSC_6735`、`DSC_6796` 等来源；`DSC_6796` 同时触发 noise ratio 失败，不能只用接缝平滑解释。
- 两套 owner sharpness coverage 均远低于 0.99，分别为 0.9114 和 0.8735；在这个证据状态下修改 seam/tone 不能合理宣称达到 D3。
- 25% 诊断画布和预览坐标不能外推到 scale=1；本任务没有原生输出，也没有用户 D4/D5。

## 最小实验边界

下一轮若进入代码，必须隔离为单一流式 mosaic 变更，并先在保存的失败 ROI 上做离线对照：

1. 先修来源/排序合同观测，确保 excluded source 明确进入运行记录；不能用人工诊断解释覆盖 runner 的 source digest 失败。
2. 固定原始 owner/coverage 与输入 SHA，只比较一个局部几何或 owner fallback 候选；不得同时改 scale、boundary、noise、owner 阈值。
3. 对每个候选保存 before/after 的 local-scale、boundary、noise、owner coverage、source path、detail donor 和 failure chain；不接受只看 QG 总 verdict。
4. 候选必须保留无支持区域和软区排除，不得把透明/未解析点补成有效覆盖；失败时回退到原路径。
5. 先用 0.25 保存诊断验证，再决定是否值得跑原生；0.25 通过不能外推 D2/D3。

本次不进入生产修复，也不勾选任务11。任务11的下一步依赖一个明确的单变量几何/owner 假设和可复现 ROI，而不是继续扩大素材运行规模。

## 补充实验：保存 owner grid 的 3×3 fallback 审计

使用正确的 test-only `RAW_EDITOR_STREAMING_MOSAIC_DIAGNOSTICS_DIR` 运行产物 `output/acceptance/task-11/qixu-1-diagnostics`，按 report 的 world origin `(-4581,1)` 和 metadata 的 analysis scale `3.2338867` 将失败点映射到 owner/coverage 平面。实验只读取 3×3 owner stencil，要求 Coverage，排除 owner=0，不扩大搜索范围。

- 14 个 scale 失败采样点中，**0 个**有替代 owner 候选；所有 3×3 邻域是同一 owner。
- 3 个 boundary 失败点中，**3 个**有混合 owner 邻域；它们只能作为几何候选，不能直接替换或视作修复。
- owner coverage 失败点位于无 Coverage 邻域，**0 个**可用 fallback。

因此当前最小 fallback 假设没有足够收益证据：scale/owner 失败不能靠同一 3×3 邻域补救，boundary 混合点需要独立 source resampling 和 QG 重算。审计脚本：[qixu_owner_fallback_audit.py](../../scripts/stack-lab/qixu_owner_fallback_audit.py)；报告：`/Users/dp/Desktop/raw-editor/output/acceptance/task-11/qixu-owner-fallback-audit.json`。本次不改生产 owner 选择。

## ownership feather test-only A/B（2026-11-19）

在冻结 `base-2026-11-12`、祁序1 全量 75 输入、scale=0.25、exclusive 槽下，固定其余路径，只改变 `RAW_EDITOR_FLAT_OWNERSHIP_FEATHER`：默认 `0.65`、实验 `1.0` 和 `1.5`。

- 默认：6 个 `local_scale_pixel_ratio=0` 失败、3 个 boundary 失败、owner coverage `0.9113826565`。
- feather `1.0`：仍是 6 个 local-scale 失败；boundary 失败降为 1 个但测值为 4px；owner coverage 提到 `0.9153221017`；QG 仍 fail。
- feather `1.5`：仍是 6 个 local-scale 失败；boundary 2 个 4px 失败；owner coverage 回落到 `0.9143601652`；QG 仍 fail。

feather=1.0 只对 seam/owner 有局部改善，不能解释或修复几何尺度失败，不进入生产默认，也不授予 D2/D3。机器摘要：[qixu-task-11-feather-audit.json](../benchmarks/qixu-task-11-feather-audit.json)。

下一步转向 local-scale ratio=0 的 source-to-world 几何/采样根因；不再继续调 feather，也不自动应用 3×3 owner fallback。

## factor source reconciliation（2026-11-22）

复核发现历史 legacy geometry A/B 的 source-factor 绑定有误：`qixu-task-11-geometry-sensitivity-audit.json` 记录 `DSC_3867.NEF` 的理论 factor 为 `1.2166592079`，而 `DSC_3882.NEF` 的 source-specific factor 为 `1.2546019382`；历史 A/B 却把 `1.2166592079` 应用到了 `DSC_3882.NEF`。因此该历史 A/B 不能验证 DSC_3882 的正确几何假设，也不进入生产或D3结论。

机器摘要：[`qixu-task-11-factor-source-reconciliation-r1.json`](../benchmarks/qixu-task-11-factor-source-reconciliation-r1.json)。若再次运行，只能固定 DSC_3882 与其 source-specific factor `1.2546019382` 做单变量 test-only 对照；不扫描多因子、不改阈值/bounds/fallback，也不把旧 A/B 结果外推。

## source-to-world geometry sensitivity（只读）

对祁序1 保存的 14 个 local-scale failure records 重算 source-to-world sensitivity：6 个实际低于 0.95 的点需要理论 isotropic factor `1.095009–1.254602` 才能到 0.95；其余 8 个 ratio/median 失败点不能从 `0` 的测量值倒推倍率。该审计只保存失败 world 坐标、owner source id、原始 homography 和理论 factor，不修改 pose、阈值或像素。

机器摘要：[qixu-task-11-geometry-sensitivity-audit.json](../benchmarks/qixu-task-11-geometry-sensitivity-audit.json)。这些 factor 只是几何假设证据，不能直接套用到生产；下一步若继续，必须固定单一 station/source 假设做 test-only bounded A/B，并同时复核 closure、boundary、owner、density 和来源合同。

## legacy source-scan geometry A/B（已停止）

为覆盖祁序实际使用的 legacy source-scan 路径，加入了仅 test binary 的 source pose hook，对 `DSC_3882.NEF`（source id=39）施加审计得到的 factor `1.2166592079`。祁序1 75 输入、scale=0.25 独占运行中，1 个 local-scale ratio 从 `0` 提到 `0.01953125`，boundary 失败从 3 个降到 2 个；但 owner coverage 从 `0.9113826565` 降到 `0.8852725022`，QG 仍 fail，来源计数与 legacy runner 观测缺口也仍在。

该单源 pose 假设不能推广；实验 hook 已从主线移除。机器摘要：[qixu-task-11-legacy-geometry-ab.json](../benchmarks/qixu-task-11-legacy-geometry-ab.json)。

## 单源 source resampling bounded A/B（2026-11-19）

learned provider 候选虽能在独立 ONNX 环境输出匹配，但没有 descriptors→Rust `FeatureProviderResult` adapter，也未通过 Rust geometry gate；按任务边界转入一个不改 pose 的单变量实验。使用同一 `task-11-legacy-geometry-r1` test binary、同一 75 张输入、scale=0.25、`exclusive` 和 streaming diagnostics，仅对已知失败源 `DSC_3882.NEF`（source id=39）设置 `RAW_EDITOR_TEST_SOURCE_RESAMPLE_DIVISOR=1.25`。控制组不设置该环境变量。

- 控制组：[qixu-1-source-resample-dsc3882-control-r1](/Users/dp/Desktop/raw-editor/output/acceptance/task-11/qixu-1-qixu-1-source-resample-dsc3882-control-r1)，实验组：[qixu-1-source-resample-dsc3882-r1](/Users/dp/Desktop/raw-editor/output/acceptance/task-11/qixu-1-qixu-1-source-resample-dsc3882-r1)。两组 binary SHA、输入 SHA 和参考摘要一致；内存门均通过。
- 两组 QG 均 fail：local-scale-median 为 `not_applicable`，local-scale pixel ratio 仍 `6/8` 个零值，boundary 仍 `3` 个失败，owner coverage 由 `0.9113826565` 变为 `0.9140979335`，detail repairs `30→34`；来源计数保持 `75/72/70`，closure 仍 `not_run`。
- 实验组 wall `983.44s`、峰值 RSS `12,796,116,992B`；控制组 wall `1046.21s`、峰值 RSS `13,238,452,224B`。TIFF/Stack_Report 仍被来源合同和导出合同拒绝，不能把资源差异解释为质量收益。

机器摘要：[qixu-task-11-source-resampling-ab.json](../benchmarks/qixu-task-11-source-resampling-ab.json)。结论是拒绝把单源 resampling 推广到生产；不改生产 resampling、owner 或阈值，也不把 0.25 结果外推到原生 D2/D3。

## 联合 geometry + owner A/B（已否定）

将 DSC_3882 factor `1.2166592079` 与 feather `1.0` 组合后，local-scale ratio 的一个点仍为 `0.01953125`，但 owner coverage 只有 `0.8850510328`，boundary 仍有两个 4px 失败，QG 继续 fail。该联合假设不能推广；停止继续扫描 pose/feather 参数。

机器摘要：[qixu-task-11-joint-geometry-owner-ab.json](../benchmarks/qixu-task-11-joint-geometry-owner-ab.json)。

## 正确 DSC_3882 source-specific factor bounded A/B（2026-11-22）

按 source reconciliation 对账，重新使用 `DSC_3882.NEF`（source id=39）对应的唯一 test-only factor `1.2546019381835418`，控制组不设置几何覆盖，实验组只对该源应用居中各向同性倍率；其余输入、scale=0.25、线程、streaming mosaic、QG 与阈值保持不变。二进制为 `candidate-release-binary`，SHA-256 `624e56a783f39d858e93825288a60d8d85f6882f40dee1761fa97639e64b38e`。

- 控制组输入 75、排序 72、streamed 70；local-scale pixel ratio `6/8` 失败、boundary `3/15` 失败、owner coverage `1/1` 失败；峰值 RSS `17,263,673,344 B`，QG fail。
- 实验组输入 75、排序 72、streamed 69；local-scale 仍 `6/8`，boundary 变为 `4/16`，owner coverage 仍 `1/1`；峰值 RSS 增至 `19,126,435,840 B`，QG fail。实验组比控制组额外缺少 `DSC_3884.NEF`，因此来源参与不一致，不能视作 clean A/B。
- 结论：正确的 source-specific factor 没有改善 local-scale，边界和资源指标变差；仅保留为诊断证据，拒绝推广到生产，也不授予 D3。停止继续扫描 factor；若再运行，先修复并冻结输入/排序/source contract，使两组参与源完全一致。

机器摘要：[`qixu-task-11-geometry-factor-correct-ab-r1.json`](../benchmarks/qixu-task-11-geometry-factor-correct-ab-r1.json)。

## 输入、排序与 soft-frame source contract 冻结（2026-11-22）

对两次已有日志做 `--reuse-artifacts` 重评，没有启动新的 RAW 渲染。此前误报的三张未对齐源已在 qixu-1 manifest 登记为 `expectedSkipped`：`DSC_3842.NEF`、`DSC_3910.NEF`、`DSC_3911.NEF`。排序合同固定为 72 项，输入总数 75，输入清单 SHA、ordered basenames SHA、baseline source-alignment SHA 和全源分类均写入冻结摘要。

- 控制组合同通过：72 ordered、70 streamed，实际 soft-skip 与冻结值一致：`DSC_3879.NEF`、`DSC_3907.NEF`；其中 `DSC_3907.NEF` 只作为 detail-only donor。
- 实验组合同仍失败：72 ordered、69 streamed，实际额外 soft-skip `DSC_3884.NEF`，并因此少一张 streamed source。`DSC_3884.NEF` 不能登记为静态 `expectedSkipped`，因为控制组排序中包含且实际 streamed；把它加入静态排除会掩盖变量导致的参与变化。
- 阶段性决策（固定参与 hook 前）：source contract 已冻结，但 factor 实验尚不具备 clean A/B 条件；因此先不重跑。后续若继续，必须增加 test-only 的固定 soft-skip/参与集合，使控制组与实验组的 streamed 集合完全一致；下一节记录了该补充实验。生产阈值、bounds、fallback 或 pose 不变。

机器摘要：[`qixu-task-11-source-contract-freeze-r1.json`](../benchmarks/qixu-task-11-source-contract-freeze-r1.json)。

## 固定 soft-skip 参与集合后的单因子 bounded A/B（2026-11-22）

为消除前一轮 `DSC_3884.NEF` 动态 soft-skip 造成的参与差异，新增 test-only `RAW_EDITOR_TEST_FIXED_SOFT_SKIP`，控制组和实验组都固定排除 `DSC_3879.NEF`、`DSC_3884.NEF`、`DSC_3907.NEF`；缺名会 fail-closed。该 hook 只在 test binary 生效，生产 soft-frame survey 不变。两组重新使用相同 75 输入、72 ordered、69 streamed 和相同 ordered names SHA；实验组只对 `DSC_3882.NEF` 应用 source-specific factor `1.2546019381835418`。

- 控制组 source contract 通过；local-scale pixel ratio `6/8`、boundary `3/15`、owner `1/1` 失败；峰值 RSS `19,018,563,584 B`，memory gate 通过。
- 实验组 source contract 通过；local-scale 仍 `6/8`，boundary 变为 `4/16`，owner 仍 `1/1`；峰值 RSS `18,997,837,824 B`，memory gate 通过；日志确认 test-only geometry factor 已应用。
- 结论：在参与集合完全一致的 clean A/B 中，正确 factor 仍未改善 local-scale，boundary 反而变差；拒绝生产推广，不修改阈值、bounds、fallback 或 pose，也不授予 D3/native 验收。停止继续扫描 factor。

机器摘要：[`qixu-task-11-geometry-factor-fixed-soft-ab-r1.json`](../benchmarks/qixu-task-11-geometry-factor-fixed-soft-ab-r1.json)。

## 最终收口与任务交接

clean A/B 已证明 source-specific factor `1.2546019381835418` 不能改善祁序1的 local-scale：两组均 `6/8` 失败；boundary 从 `3/15` 变为 `4/16`，owner coverage 仍 `1/1` 失败。两组 source contract 和 memory gate 均通过，因此结果可解释，但 QG 仍 fail。

任务11到此收口：停止继续扫描 factor，不把 test-only fixed soft-skip 或几何 hook 写入生产，不修改阈值、bounds、fallback 或 pose，不授予 D3/native 验收。下一项转任务12枫鹰，保留任务12既有诊断证据和同样的 no-promotion 约束。

交接摘要：[`qixu-task-11-to-task-12-handoff-r1.json`](../benchmarks/qixu-task-11-to-task-12-handoff-r1.json)。
