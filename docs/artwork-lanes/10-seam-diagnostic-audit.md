# 任务10：书画接缝诊断回归核对

- 更新：2026-11-18；状态：诊断回归完成，真实接痕语义验收仍未通过；分支 `lane/task-10/seam-diagnostic-audit`；起点 v1 `e6be5268`。
- 范围：BR.5–BR.7 的现有 test-only/前端诊断链路回归；本次只记录证据，不改像素、不启用自动修复、不改变阈值。

## 回归结果

| 检查 | 结果 |
| --- | --- |
| `npm run brightness-map:check` | pass |
| `npm run adaptive-brightness:check` | pass：subdivision、continuity、protection、hue、alpha、export parity |
| `npm run seam-candidates:check` | pass：steps、curves、ramps、scratches、chroma、alpha |
| `npm run seam-texture:check` | pass：曝光不变性、方向、位移、低纹理、缺失数据 |
| `npm run native-seam-texture:check` | pass：随机绢纹、增益不变性、方向、缺失支持 |
| `npm run native-seam-profile:check` | pass：偏移/斜阶、留出带、控制、渐变、细线、聚合 |
| `npm run native-seam-path:check` | pass：弱漂移/斜路径、绢纹、线/渐变、部分/符号冲突、无效控制 |

这些是合成/局部诊断回归，不是跨图片召回率，也不是原生整图质量验收。

## 当前可证明范围

- 候选检测、法线采样、源尺寸局部采样、多区域纹理、路径聚合和切图/卸载失效逻辑均有单元或组件级保护。
- 所有诊断结果仍是候选、位置、纹理证据或 `insufficient` 状态；没有任何诊断分支写入预览/导出像素。
- UI 仍明确标记“接缝诊断（实验）”“仅诊断”；排除候选只改变诊断状态，不把像素或 QG 结果改成通过。

## 未通过项与下一步

BR.5–BR.7 仍不能勾选：现有三张真实图没有人工标注的真实接痕、装裱/阴影/笔墨/正常纹理负例和独立图片集；历史实测仍有弱缝漏检、画面轮廓误检和纹理证据不足。四景山水横向弱缝仍可能漏检，候选增加不能解释为接痕已消除。

自动修复继续保持关闭。下一次真实验收必须提供：

1. 每张图的真实接痕 polygon/中心线、损伤类型和允许误差；
2. 同图的正常笔墨、装裱边、反光、阴影、绢纹负例；
3. 未参与调参的独立图片集；
4. 位置误差、漏检、误检、纹理连续性和修复前后像素/来源/QG 对照。

在这些证据出现前，不把诊断通过写成 BR.5/BR.6，也不设计或启用 BR.7 自动修复。
