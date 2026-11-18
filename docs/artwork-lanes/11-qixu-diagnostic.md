# 任务11：祁序到 D3 诊断边界

- 更新：2026-11-18；状态：诊断完成，修复未开始；分支 `lane/task-11/qixu-diagnostic`；起点 v1 `4d7d33aa`。
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
