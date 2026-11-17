# 书画复看队列

[返回任务卡说明](README.md) · [当前任务](../../.kiro/specs/layered-camera-group-focus-stitching/tasks.md)

当前已登记待看：**1 / 3套**。活跃批次：**R20261118-1**。历史已关闭批次：**0**。

这里只记录实际提交给用户的复看包；模板、准备中的素材、D3机器验收记录不自动算作已提交。队列数量不代表D4已完成：任务6的兰亭r2整图与6对原生裁片已提交，尚待用户确认；阆苑/文苑10张结论不能外推全量。相关机器验收见 [任务2记录](2-run-matrix.md)及[任务6记录](6-paper-seams.md)，机器pass和agent看图均不代替用户D4。

## 当前待看

| 批次 | 素材与输入范围 | 成片版本/TIFF SHA | 入队前等级 | 状态 | 整图路径 | 关键原生裁片及坐标 | 卡片/证据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R20261118-1 | 兰亭45张完整/scale=1/显式取景 | task-6-paper-r2 / `29852099097a03e489d5addc0b04fca53860ef82e2a62ca3c08f0cffbda3ff02` | D3（v4严格保护及中等层5/5已过） | 待看，不计D4 | `/Users/dp/Desktop/raw-editor/output/acceptance/task-6/review-r2/after-whole.jpg` | C1–C6各512×512、RGBA16；表及逐像素证明见下 | [任务6](6-paper-seams.md) / [复看包](/Users/dp/Desktop/raw-editor/output/acceptance/task-6/review-r2/README.md) |

## 批次 R20261118-1：兰亭纸面接缝修复

- 提交日期：2026-11-18；整理人/L0：Codex；状态：活跃，1套；跨批次待看1/3。
- 输入：兰亭完整45 JPEG；输入清单SHA `00af0b142f22ef38138b078540b8156f75b5220557a93a5310d20caad120f4b1`；scale=1，显式取景。
- 冻结：task-6-paper-r2，commit `72af6ea0bf011b3aea1d59a0e4cfe6570ac1e22b`，binary SHA `de905b0e6e1f3f620a88db2449a6df4f1672212d8e8f7fdc71862960f93db522`；QG pass/8ROI、unresolved0、RSS `11103862784` B。
- 成片：`/Users/dp/Desktop/raw-editor/output/acceptance/task-6/lanting-r2/lanting/run/lanting-s1-task-6-r2.tiff`；31090×11008 RGBA16、sRGB；TIFF SHA `29852099097a03e489d5addc0b04fca53860ef82e2a62ca3c08f0cffbda3ff02`。取景在source-union内(2606,874)，source-union原点(1,1)，streaming canvas原点(2607,875)。
- 整图与全部原生裁片：[打开复看包](/Users/dp/Desktop/raw-editor/output/acceptance/task-6/review-r2/README.md)。候选整图 `/Users/dp/Desktop/raw-editor/output/acceptance/task-6/review-r2/after-whole.jpg` SHA `27b5e096c4d09e124aefcf7fb41c1d55600b8b540e07871e2c907da8dce750c4`；仅缩略预览，不能代替下面的1:1裁片。
- 对照：I1/任务4字节一致TIFF SHA `22b5a3717a2411f728b84d0ec1e4a1f852485abd38de43fc027610f8fedf35ac`；原始只读参考 `/Users/dp/Downloads/书画拼图/兰亭/DSC08854_FocusStack.jpg` SHA `9981d5ff735ffaed21d0908c5165a984aef8c71e580dcff566fee92d57d845c2`。
- 需看：整图纸面和色调；C1–C3来源边界与印章；C4细笔画；C5印章/织纹；C6真实硬件反光保留。机器纸面P95 4→3，max7→6，>6处2→0；纸面RGB残差仍6/1/1。画面MAE artwork4.05→3.99，bottom5.41→5.23，wall6.59→6.66，rail5.53→5.57，非所有区都下降。
- 原始r2/v3拒绝只因预期TIFF摘要变化，记录未改写；本素材v4严格保护另存 `/Users/dp/Desktop/raw-editor/output/acceptance/task-6/r2-v4-strict.json`，通过。v4未改其他素材或共享质量阈值，旧sharpness floor未重设。集成中等层5/5通过；兰亭同冻结二进制两次TIFF及纸面审计一致。

| 裁片 / output x,y,w,h | 基线原生TIFF（来源SHA见上） | 候选原生TIFF（来源SHA见上） |
| --- | --- | --- |
| C1-paper-owner14-9 `[17354, 8205, 512, 512]` | `/Users/dp/Desktop/raw-editor/output/acceptance/task-6/review-r2/before-001.tiff` SHA `07ffcdcb0a188c15891a1d0a6164ed423a81e3e03061f3349f83466bede2e59e` | `/Users/dp/Desktop/raw-editor/output/acceptance/task-6/review-r2/after-001.tiff` SHA `5ffee801b2a0b4c4bfea6dc45fc92a5fa94377fb20750c75434794cad33a5f44` |
| C2-paper-left-seals `[1809, 7719, 512, 512]` | `/Users/dp/Desktop/raw-editor/output/acceptance/task-6/review-r2/before-002.tiff` SHA `4bc8f13ef4a53b79abd660e67bb31480103fa9dcf16a7c60e4e1206479636cac` | `/Users/dp/Desktop/raw-editor/output/acceptance/task-6/review-r2/after-002.tiff` SHA `c77f80ecd2d7b865d0b878ee9d46ad66d7a4aeeae18207268ec0f72711ffeaf9` |
| C3-top-paper-ink `[26583, 2382, 512, 512]` | `/Users/dp/Desktop/raw-editor/output/acceptance/task-6/review-r2/before-003.tiff` SHA `aecb80ffdfd483e1e5ae151165b6aae5ac6d8b81d05036b4e10a73f57ea8ac6d` | `/Users/dp/Desktop/raw-editor/output/acceptance/task-6/review-r2/after-003.tiff` SHA `2aa36a884ccf328e923d367cfd02deb3a091ad2bfecb2944e0859d3d24296fd8` |
| C4-central-ink `[14925, 5051, 512, 512]` | `/Users/dp/Desktop/raw-editor/output/acceptance/task-6/review-r2/before-004.tiff` SHA `86a32c9d18a7846e29ad5b7dc1cdaf09587363860c2f72045d2390917c678754` | `/Users/dp/Desktop/raw-editor/output/acceptance/task-6/review-r2/after-004.tiff` SHA `1940e2fd17317e8d5ca8eef31da5e6e929103b1ff90ff60e3c038c9572d22ddb` |
| C5-right-seals `[29954, 6567, 512, 512]` | `/Users/dp/Desktop/raw-editor/output/acceptance/task-6/review-r2/before-005.tiff` SHA `4fd6ac07c25a337980b683251ad8109493292a60ad5fa9c86e17a3e38459414c` | `/Users/dp/Desktop/raw-editor/output/acceptance/task-6/review-r2/after-005.tiff` SHA `fbac0ffeca704b6abbd8e54db28f8f41a5a7bdf34ad137b2fa711a40dd11567c` |
| C6-real-bottom-reflection `[17202, 10206, 512, 512]` | `/Users/dp/Desktop/raw-editor/output/acceptance/task-6/review-r2/before-006.tiff` SHA `cda35a3e1a60e3cb87ee6021d74107d4014c7c1927ca27dd5bba54b7a1a95346` | `/Users/dp/Desktop/raw-editor/output/acceptance/task-6/review-r2/after-006.tiff` SHA `e44355eff06bead69732f2fcd748fbd51c8f0210aa8e37252d99b1ca6f1b8c68` |

- 所有PNG路径、SHA、逐字节RGBA证明和ICC摘要：[manifest.json](/Users/dp/Desktop/raw-editor/output/acceptance/task-6/review-r2/manifest.json)。PNG与裁片均512×512、16位、无重采样；1-based裁切坐标与0-based窗口已由独立解码校验。
- 用户记录：尚无。整图/关键裁片/颜色/细节/接缝均待用户确认；不自授D4；未回复不视为通过。
- Agent初看：整图与C1–C6 before/after均已打开；这些位置未见明显新光晕、笔画断裂或真实反光丢失；这不是用户确认。
- 结论适用范围：仅上述输入与r2 TIFF，不覆盖其他素材、任意未取景模式或桌面打包；中等层5/5已通过，由L0本地集成；视觉结论另记。

## 入队与关闭规则

- 全部活跃批次合计最多3套；`待看`、`待复看`、`待补充确认`均计入待看数，不是每批各限3套。同一输入范围的多版本对照算一套，不同全量/子集或输入摘要不能混作同一次验收。
- 通常在D3后入队。必须有已核对存在的整图、关键1:1原生裁片和明确问题；写绝对路径、SHA及对应成片版本。1200像素预览或放大图不能代替原生裁片。缺材料先标`准备中`留在诊断卡，不占当前待看表、不向用户声明可复看。
- 不到D3而需要用户协助定位时明确标为`诊断意见（不计D4）`；实际请求用户看图时同样计入3套上限。机器拒绝、内容不适用和证据不足不能靠用户目视抹去。
- 入队时更新顶部总数与活跃批次，在下方复制批次模板填写真实数据；当前表每套一行，模板中的占位行不计数。满3套后其余停在D3，转做下一套诊断。
- 用户只确认整图、只确认颜色、未看关键裁片或范围含糊时记`待补充确认`，不记D4。保存用户原话、日期和对应版本；未回复不能视作同意。
- 用户明确通过整图及关键裁片时，该范围可记D4；要求修复则记录拒绝位置并回到对应区域组。用户明确暂缓可关闭为`暂缓`。这些记录都移入历史，不删除旧版本或失败。
- 换输入范围、改成片像素或重新取景后须重新确认，不能挪用旧D4；同套新版替换时旧包归档为`已替换（未确认）`并链接新批次，仍维持跨批次总数≤3。D5另外核对复现与资源合同，不由队列自动授予。

## 批次记录模板

复制围栏内模板后填写；不要把占位路径当作已检查的文件。每套素材都要单独记录，不以一条总评推定整批通过。

```markdown
## 批次 R<日期>-<序号>：<目的>

- 提交日期：<日期>；整理人/L0：<填写>；状态：<活跃/已关闭>。
- 本批套数：<N>；跨批次待看总数：<N，必须≤3>。

### <dataset>：<全量N张/具体子集>

- 诊断卡：<仓库相对链接>；输入manifest/清单SHA：<填写>。
- 当前等级与范围：<D3或诊断意见（不计D4）>；机器结果：<QG/闭环/来源/资源、证据路径>。
- 冻结二进制：<label、SHA>；运行版本：<label、提交、execution/analysis/Stack_Report路径>。
- 成片：<原生TIFF绝对路径、尺寸、SHA、scale、crop/原点>。
- 整图：<实际预览绝对路径、尺寸、SHA；注明缩略预览用途>。
- 对照：<旧版本/只读参考路径和SHA；没有参考则明确写无>。
- 需看问题：<位置、观察内容、已知反光/色阶、需要用户确认的范围>。

| 原生裁片绝对路径 | 来源TIFF SHA | 取样方式/坐标域 | x,y,w,h与crop原点 | 需看内容 |
| --- | --- | --- | --- | --- |
| <逐一填写真实路径及裁片SHA> | <填写> | <1:1无重采样、output或source> | <填写；保证能对应整图> | <笔画/印章/软区/owner边界/色调等> |

- 用户记录：<日期、消息/讨论定位、原话；尚未回复就写尚无>。
- 分项确认：整图<填写>；关键原生裁片<填写>；颜色/细节/接缝<填写>。
- 结论：<待看/待复看/待补充确认/通过D4/需修复/暂缓/已替换（未确认）>。
- 结论适用范围：<具体输入与成片版本>；仍有争议/不覆盖的范围：<填写>。
- 后续：<修复区域组与任务、新批次链接或D5缺失证据>。
- L0关闭日期与理由：<填写；不得用机器pass替代用户结论>。
```

## 历史批次

暂无。关闭批次移到此处，保留原版本、完整路径、用户原话及后续链接；关闭后更新顶部和当前待看表。
