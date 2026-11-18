# RAW Editor 项目进度与任务总览

更新日期：2026-11-18（新增任务1–4基础设施、任务5全量诊断及任务6兰亭接缝机器验收；图像验收状态仍以既有证据为准）。**总入口。本文件只记录整体状态、验收证据和文档导航；任务按类别各在一个文件里维护，见[§2 任务在哪里](#next)。书画堆栈开发从[tasks.md](../.kiro/specs/layered-camera-group-focus-stitching/tasks.md)开始。** 功能文档保存方案、接口、实现边界与验收证据；历史日志不再充当另一份当前任务表。

状态覆盖堆栈、分区亮度/接缝诊断、渲染与内存、色彩、工程与发布。未有验收记录的产品能力不推定为完成，也不按勾选数量计算项目完成率。以下是已有证据的汇总；2026-11-12的I1集成验收见[§7.1](#acceptance)及[I1机器摘要](benchmarks/i1-integration-baseline.json)。

[当前状态](#overview) · [任务在哪里](#next) · [验收基线](#acceptance) · [专项文档](#documents)

<a id="overview"></a>

## 1. 当前整体状态

| 模块               | 已完成或已具备                                                                                                                                                       | 尚未完成 / 不能据此宣称                                                                         |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 堆栈路径与缓存     | P0路径观测、singleton scan修复；统一SourcePreflight；有界source/ROI/native fusion复用；默认512MiB/1GiB配置下9张回归通过；PF.6冻结v34/v35的ABBA＋BAAB八轮计时归因完成；I1起导入前扫描拍摄参数，多文件夹/多次拍摄/参数不一致用逐帧曝光平面，参数一致统一调整 | 历史v35慢跑保留；八轮未复现稳定的版本级耗时回退，更大规模及跨素材性能尚未通过                   |
| 9张NEF正对照 | v28冻结基线；I1g同二进制冷/热质量、来源和默认512MiB/1GiB预算回归通过，TIFF保持v28、RAW18/fusion6 | 仅DSC6479–6487，不等于睡觉45张全量通过；与枫鹰并发的I1f热跑出现站点关系差异，独立重跑通过，确定性待N8.2 |
| 文苑10张           | v35联合原生几何、闭环、覆盖、QG和原生验收通过；v44受影响素材复验通过且TIFF摘要不变                                                                                   | 文苑126张全量尚未验收                                                                           |
| 阆苑10张           | v44三条独立关系、输出像素闭环、QG和16位TIFF完整通过；同二进制重复运行输出摘要一致                                                                                    | 输出P95距3px门槛约0.012px；阆苑84张全量及跨素材稳定性尚未验收，PS验证域仅27.4%                  |
| 兰亭45张 / 马远PSD | 任务6 r2：45张显式取景、31090×11008 RGBA16；QG pass/8ROI、unresolved0，跨owner纸面P95 4→3、max7→6、>6处2→0；同冻结二进制两次TIFF相同；v4加严保护与中等层5/5通过。I1/v63历史见§7.1；马远14个PSD的导入/10%回归历史保留 | 兰亭仍D3；R20261118-1整图及6对原生裁片已提交，未作用户D4/D5确认。其他legacy/未取景路径及PSD充分QG证据未外推 |
| 分区亮度           | 6×4手动场、自适应校正、区域保护、预览/导出模型复用及交互已实现                                                                                                       | 人像、逆光、扫描件、正常无缝素材等通用实图验收未完成                                            |
| 接缝诊断           | 候选、源尺寸局部定位、同侧纹理对照、弱响应路径及多区域复核已实现                                                                                                     | 真实接痕识别和自动修复未验收；自动修复未启用，诊断不改像素                                      |
| 渲染与色彩         | 四级渲染、共享缓冲/条带导出、ICC契约；单个α7R V样片的范围化CPU基线                                                                                                   | 完整RAW开发、部分编码/蒙版仍有整图峰值；应用内60MP GPU与多场景色彩验收不全                      |
| 工程与发布         | 大规模模块拆分；2026-11-03 macOS打包及单图桌面冒烟；基础版本`base-2026-11-12`的Rust库测试1093通过/0失败/44忽略；runner/兰亭保护Node测试10/6通过                                                        | 后续修改未重做完整桌面集成；剩余长函数/组件、依赖方向、研究契约隔离、桌面矩阵与跨平台发布未完成 |

状态解释：`[x]`只表示条目明确写出的范围完成；`[ ]`包括部分实现、待验证和未实施，具体差距写在条目中。已有功能、测试通过、生成文件和真实验收通过分别记录。

<a id="next"></a>

## 2. 任务在哪里

**每类任务只在一个文件里维护和勾选。** 要开始开发，先看第一行。

| 想做什么 | 看哪个文件 | 内容 |
| --- | --- | --- |
| **书画堆栈开发（当前主线）** | [tasks.md](../.kiro/specs/layered-camera-group-focus-stitching/tasks.md) | 产品分析、当前缺陷、改进计划、按批任务清单、储备任务、并行规则与回归矩阵 |
| 工程、代码结构、打包发布 | [tasks/engineering.md](tasks/engineering.md) | ENG.1–ENG.7，原N0–N2代码结构任务（已完成10项，未完成8项） |
| 分区亮度与接缝诊断 | [tasks/brightness-seam.md](tasks/brightness-seam.md) | BR.1–BR.8（已完成3项，未完成5项） |
| 渲染、内存与色彩 | [tasks/render-color.md](tasks/render-color.md) | RM.1–RM.7、CL.1–CL.5（已完成3项，未完成9项） |
| 查堆栈旧条目的完整说明 | [tasks/stack-history.md](tasks/stack-history.md) | 原PF、N3–N9全文和历次验收记录，只读；当前状态以tasks.md为准 |
| 任务卡、诊断卡与用户复看 | [artwork-lanes/README.md](artwork-lanes/README.md) / [review-queue.md](artwork-lanes/review-queue.md) | 模板、新素材接入；当前待看数量及按批材料只在复看队列登记 |
| 看验收证据 | 本文件[§7](#acceptance) | 最新真实结果、数据集矩阵、验收规则 |

本文件只记录整体状态、验收证据和文档导航，不再存放待办。2026-11-12前本文件§3–§6的任务条目已按类别原样迁入上表各文件；更早的83项勾选见[历史快照](archive/focus-stack-tasks-2026-11-04.md)，原推进顺序表见[归档的书画推进计划](archive/artwork-stack-plan-2026-11-12.md)末尾。

<a id="stack"></a>

## 3. 堆栈任务

已迁至[tasks.md](../.kiro/specs/layered-camera-group-focus-stitching/tasks.md)；原PF与N3–N9条目全文见[tasks/stack-history.md](tasks/stack-history.md)，原N0–N2见[tasks/engineering.md](tasks/engineering.md)。

<a id="n0"></a><a id="n1"></a><a id="n2"></a>

- N0–N2（基线保护、职责拆分、测试隔离）：[tasks/engineering.md](tasks/engineering.md#n0)

<a id="n3"></a><a id="n4"></a><a id="n5"></a><a id="n6"></a><a id="n7"></a><a id="n8"></a><a id="n9"></a>

- N3–N9（观测、几何、owner、seam/tone、QG、确定性、算法研究）：[tasks/stack-history.md](tasks/stack-history.md#n3)

<a id="brightness"></a>

## 4. 分区亮度与接缝诊断

已迁至[tasks/brightness-seam.md](tasks/brightness-seam.md)。

<a id="render"></a>

## 5. 渲染、内存与色彩

已迁至[tasks/render-color.md](tasks/render-color.md)。

<a id="engineering"></a>

## 6. 工程、集成与发布

已迁至[tasks/engineering.md](tasks/engineering.md)。

<a id="acceptance"></a>

## 7. 当前验收与素材范围

### 7.1 最新真实结果

| 范围                     | QG与最终结论                                          | 关键数据                                                                                                           | 证据                                                                   |
| ------------------------ | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| 任务7阆苑置换检验（2026-11-18）（2026-11-18） | test-only离线实验完成，不改生产路径或门槛 | 84张只读代理、76条真实跨组边、1000次非相邻置换；随机NCC P99=0.976218，真实39/76高于；0–2原生证据与代理分布不在同一统计域，未替换门槛 | [任务7摘要](benchmarks/langyuan-task-7-permutation.json) / [任务卡](artwork-lanes/7-permutation.md) |
| 任务6兰亭纸面接缝（2026-11-18） | 规定实现和机器验证通过，本地合入；用户复看待确认，仍D3 | 同位重叠图/渲染基拟合、holdout与8%界限；纸面P95=3、max6、>6=0，QG pass/8ROI；输入/owner/coverage/canvas保持；Rust1098/0/44、Node19、quick7、medium5/5；同二进制两次TIFF相同，中等最大RSS10.726GiB；仅本素材v4及矩阵更新，旧锐度floor/其他素材保持；复看队列1/3 | [任务6摘要](benchmarks/lanting-task-6-paper-seams.json) / [任务卡](artwork-lanes/6-paper-seams.md) |
| 任务5全量摸底（2026-11-18） | 7次scale=0.25诊断完成，不新增原生画质验收 | 原冻结base、585输入、7张来源/失败链/资源/问题诊断卡；原始失败及观测缺口保留，最大RSS15.582GiB；输入/参考/二进制保持，生产代码和基线未改 | [任务5摘要](benchmarks/task-5-full-baseline.json) |
| 任务4线程上限 / 真实两槽（2026-11-17） | 预算缺口修复，规定回归及两槽验收通过 | 预处理池遵守Rayon=5；旧版本强制独占；真实编译+运行+排队及隔离通过；35项工具测试、快速7项、Rust1095/0/44、中等5范围TIFF均保持冻结值；最大RSS11.876GiB | [任务4摘要](benchmarks/task-4-two-slots.json) |
| 任务3任务卡 / 诊断卡 / 复看队列（2026-11-17） | 文档核对通过，无新图像验收 | 模板覆盖任务5的9组输出字段；跨批次待看≤3、任务3验收时已登记0/3（当前数量以复看队列为准）；17项文档条件通过，未重跑素材或授予D4 | [任务3记录](artwork-lanes/3-templates.md) |
| 任务2单次运行 / 回归矩阵 / 内存门（2026-11-17） | 工具及规定真实验证通过 | 32项工具测试；快速层7项、中等层5/5通过；TIFF保持冻结值，兰亭严格保护通过；错误SHA触发真实独占重跑并正确返回1；最大RSS10.371GiB≤24GiB | [任务2摘要](benchmarks/task-2-run-matrix.json) |
| 任务1资源槽 / 版本冻结（2026-11-17） | 工具验收通过，不新增画质结论 | 17项进程测试通过；base二进制SHA与原记录一致；真实Cargo编译冻结及3项定向冒烟通过；该次仅验收任务1 | [任务1摘要](benchmarks/task-1-resource-freeze.json) |
| base-2026-11-12（I1+codex移植） | 全部回归输出与I1相同 | 库测试1093/0/44；9张冷/热TIFF=v28；阆苑/文苑10张、兰亭45 TIFF=I1；祁序/枫鹰0.1 QG fail同I1g；兰亭保护合同v3冻结于此输出 | [基础版本摘要](benchmarks/base-p1-codex-port.json) |
| I1g兰亭45原生 | QualityGate pass/8ROI；拍摄参数一致，统一调整 | 未解析0、细节修复97；纸面残差6/1/1、阶跃P95=4、边界>6处3；四区灰度MAE 6.57/5.55/3.93/5.48（v63 7.19/12.11/4.61/6.28）；TIFF变化，目视复核待用户 | [I1摘要](benchmarks/i1-integration-baseline.json) |
| I1g睡觉9冷/热 / 阆苑10 / 文苑10 | 全部通过 | 9张TIFF=v28；阆苑/文苑quality_accepted、闭环收敛、TIFF=v47；库测试1086/0/44 | [I1摘要](benchmarks/i1-integration-baseline.json) |
| I1g祁序 / 枫鹰 0.1 | QG fail，对话框拒绝（严格合入） | 祁序owner_sharpness_coverage 0.889、边界max1.72；枫鹰DSC04916局部尺度比失败、边界P95 1.74>1.5、未解析112；输出尺寸与分支HEAD一致，自身锐度0.053/0.130（分支0.048/0.121） | [I1摘要](benchmarks/i1-integration-baseline.json) |
| v63兰亭45原生 | 原生QualityGate pass/8ROI，严格保护及CLI返回0 | 45来源/44机位；未解析118893→0；TIFF保持v60；纸面跨owner P95=3；MTF/边界内容不适用 | [v63摘要](benchmarks/lanting-v63-native-quality-acceptance.json) |
| v62睡觉9冷/热 | 原质量/来源/预算/输出断言全部通过 | TIFF保持v28，RAW18/fusion6；wall369.644/399.795s；旧基线条目保留 | [同二进制回归](benchmarks/lanting-v63-native-quality-acceptance.json) |
| v60兰亭45                | 主体色块回归/来源合同通过；严格整体质量仍拒绝         | 纸面阶跃P95 4→3；纸面裁切0；legacy insufficient_evidence/0ROI                                                      | [v60摘要](benchmarks/lanting-v60-paper-tone-baseline.json)             |
| v60睡觉9冷/热            | 全部既有断言通过，TIFF与v28一致                       | QG pass/8ROI、RAW18/fusion6、默认512MiB/1GiB；不声明稳定提速                                                       | [v60摘要](benchmarks/lanting-v60-paper-tone-baseline.json)             |
| v44阆苑10（重复2次）     | QG pass /8 ROI；三边独立闭环与原生TIFF完整通过        | 输出像素P95两次均2.987895px（3px门槛），TIFF摘要一致；余量约0.012px                                                | [v44摘要](benchmarks/langyuan-v44-joint-field-acceptance.json)         |
| v44文苑10 / 睡觉9冷热    | 文苑完整验收通过、TIFF与v35一致；9张冷/热基线全部通过 | 文苑QG 8ROI、输出P95=2.136105px；9张RAW18/fusion6、TIFF与v28一致                                                   | [v44摘要](benchmarks/langyuan-v44-joint-field-acceptance.json)         |
| PF.6 v34/v35同9张8轮交错 | 八轮QG与v28 TIFF均通过；历史慢跑未在交错均值中复现    | v34/v35 wall均值408.20/397.54s；各4轮、同版本波动较大                                                              | [PF.6归因](benchmarks/pf6-v34-v35-interleaved-attribution.json)        |
| v35文苑10                | QG pass /8 ROI；完整原生验收通过                      | 3条关系；覆盖99.472410%；wall634.717s、RSS9,101,803,520B、RAW20/fusion6                                            | [v35摘要](benchmarks/cross-station-v35-native-domain-acceptance.json)  |
| v35阆苑10                | QG pass /8 ROI；完整原生验收拒绝                      | 2条关系、no_constraints；覆盖99.441301%；wall496.666s、RSS9,656,811,520B、RAW20/fusion6；PS验证域27.3995%          | [v35摘要](benchmarks/cross-station-v35-native-domain-acceptance.json)  |
| v35睡觉9冷/热            | 质量、来源及缓存回归通过；性能无回退未确认            | 各8 ROI/8 station owner evidence、RAW18/fusion6、hits113/15；wall532.148/677.405s；RSS8,794,636,288/9,421,586,432B | [v35摘要](benchmarks/cross-station-v35-native-domain-acceptance.json)  |
| v34同9张对照             | 当时冷/热通过，原件保留                               | wall401.591/376.232s；v35耗时增加32.51%/80.05%                                                                     | [v34摘要](benchmarks/cross-station-v34-variable-scale-acceptance.json) |
| v28固定基线              | 用户确认原性能目标和既有质量门禁范围通过              | 独立分支`codex/accepted-stack-v28-baseline`，提交`504d904ff77781ac13e57f52c13fef8024b6c961`                        | [基线记录](focus-stack-accepted-baseline.md)                           |

9张固定输入为DSC6479–6487；scale=1、15897×6720 RGBA16 sRGB。v35冷/热TIFF仍与v28相同：`718a903dcd0bf102fd7ea7f2be652168ad972f18bc540f5854225ee2f1bb47e3`。11个非QG证据部分及其他质量判据不变；有效覆盖测量有意修正为物理来源域，旧矩形/全部alpha计数仍在审计中。冷/热为独立进程，未清空OS页缓存。

v35冻结候选SHA：`392f410cf9d43e2dfe3dc8a4c399fbe9997f15ae495e023dc08c4a602a3f331a`。v35摘要总状态为`partial_acceptance`，其阆苑失败是历史结果；v44已另行完成阆苑10张验收。详细可复现配置、报告哈希和本地产物路径在机器摘要中，不在总览复制每轮命令日志。

### 7.2 数据集矩阵

数量与扩展名以[数据manifest](../tests/acceptance/focus-stack-datasets.json)为准；登记输入不代表质量验收。历史张好好2已从当前manifest移除，不列为新基线。

| dataset         | 全量输入   | 当前可用结论 / 下一阶段                                        |
| --------------- | ---------- | -------------------------------------------------------------- |
| lanting         | 45 JPEG    | 45/45 legacy行为保护；完整质量/输出合同见N6.5                  |
| mayuan-psd      | 14 PSD     | 暂缓（2026-11-12用户决定）；保留单文件合成源解码与10%回归记录，不进入当前任务和矩阵 |
| langyuan        | 84 NEF     | 10张v44两次完整验收通过、输出闭环余量约0.012px；84张全量未重启 |
| wenyuan         | 126 NEF    | 10张v44复验通过且TIFF与v35一致；不能外推126张                  |
| sleeping        | 45 NEF     | 固定9张正对照通过；45张全量未验收                              |
| qixu-1 / qixu-2 | 75 /98 NEF | 合并数据集`qixu`（173张两次拍摄）I1g 0.1首次测得QG fail（owner锐度覆盖、边界描线），交L4 |
| mayuan-2        | 47 NEF     | 已登记，待建立范围化真实回归基线                               |
| fengying-main   | 63 ARW     | 整组`fengying`（110输入三文件夹）I1g 0.1首次测得QG fail（局部尺度比、边界描线、owner覆盖），交L5 |
| fengying-head   | 17 ARW     | 头部独立输入，待当前回归                                       |
| fengying-bottom | 24 ARW/NEF | 底部独立输入，不能与PS派生合成图混为原始源                     |

### 7.3 所有阶段都要保持的验收规则

1. 保留原质量门禁及拒绝语义；`not_run`、内容不适用、degraded、树图no_constraints均如实记录，不通过改阈值或抹去失败获取通过。
2. 原始素材及参考图只读；Source_RAW digest、Coverage、Ownership、provenance贯穿渲染和报告。PS参考只用于受限辅助比较，不成为生产像素或关系证据。
3. 9张基线、已冻结二进制和其他未提交修改保持独立。历史实验失败保留，不以新文件覆盖旧结果。
4. 代码变更按小职责层推进：格式/编译→定向测试→完整相关回归→数据契约→受影响真实素材及整图/原生crop。纯文档整理只检查文档完整性与链接，不伪报重跑验收。
5. 真实重任务须先领资源槽，同一时刻最多一个编译和一个运行，84/98/126张首次运行及所有性能计时独占（2026-11-11起，原“一律串行”按[tasks.md §3.1](../.kiro/specs/layered-camera-group-focus-stitching/tasks.md)改为build/run两个资源槽）；并行运行的计时标`contended`且不作性能结论；记录manifest、构建/输入/输出digest、实际输出、QG/失败阶段、来源、计时和峰值内存。临时文件不写回原素材目录。
6. 保持生产、测试支持和研究oracle边界；每个新模型有baseline、独立holdout、失败回退和资源预算。格式债务、未完成的文件长度及发布验证不由测试数量抵消。

运行入口和命令见[验收说明](../tests/acceptance/README.md)、[通用回归契约](focus-stack-generic-regression.md)、[bench使用说明](../bench/README.md)。

<a id="documents"></a>

## 8. 文档导航与归档

### 专项文档：按问题进入

| 文档                                                                                                                                                | 内容范围                                                                                           | 当前任务位置                    |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------- |
| [分区亮度与接缝诊断](adaptive-brightness.md)                                                                                                        | 手动/自适应亮度、源尺寸接缝检测、纹理与实图证据                                                    | BR                              |
| [堆栈算法](focus-stack-algorithm.md)                                                                                                                | 生产流程、适用场景、能力和限制                                                                     | PF、N4–N7                       |
| [堆栈架构](focus-stack-architecture.md)                                                                                                             | 模块入口、职责、路由与拆分边界                                                                     | N1–N2                           |
| [跨机位倍率与几何](focus-stack-variable-magnification.md)                                                                                           | 几何采样契约、联合闭环、物理覆盖及参考注册                                                         | N4、N7                          |
| [堆栈性能方案与历史记录](focus-stack-performance-optimization.md)                                                                                   | 优化机制、逐轮实验、冷/热资源明细                                                                  | PF；当前顺序只维护于本文件      |
| [开发任务 tasks.md](../.kiro/specs/layered-camera-group-focus-stitching/tasks.md) | 书画定位下的开发入口：目标与D1–D5、资源槽/分支/输出并行规则、回归矩阵、第0–3批任务清单、brief模板（原书画推进计划已并入并归档） | 全部当前任务 |
| [开源算法对比](focus-stack-open-source-comparison.md)                                                                                               | 开源堆栈/拼接项目、算法谱系、优劣、许可证与对照设计（2026-11-11联网复核，未做跨工具实测）          | PF.8–PF.11、N9                  |
| [固定基线与集成证据](focus-stack-accepted-baseline.md)                                                                                              | v28身份、保护方式和后续范围化回归                                                                  | N0、ENG、验收表                 |
| [渲染与内存](render-strategy.md) / [色彩管线](color-pipeline.md)                                                                                    | 缓冲、导出、颜色语义、基准与剩余技术约束                                                           | RM、CL                          |
| [源码布局](source-layout.md) / [样式布局](../src/styles/README.md)                                                                                  | facade与模块地图、源码约定                                                                         | N1、ENG                         |
| [需求](../.kiro/specs/layered-camera-group-focus-stitching/requirements.md) / [设计](../.kiro/specs/layered-camera-group-focus-stitching/design.md) | 堆栈硬合同与设计细节，保留原编号                                                                   | N0–N9；不另维护当前进度         |
| [验收入口](../tests/acceptance/README.md) / [通用回归](focus-stack-generic-regression.md)                                                           | 输入、运行方式、证据合同与变体矩阵                                                                 | N3、N7–N8                       |

### 历史材料：只追溯，不派发任务

- [整合前任务原文](archive/focus-stack-tasks-2026-11-04.md)：原83个勾选条目完整保留。已完成部分合并到PF/N0–N9；重复或被后续证据覆盖的待办按当前范围重新表述，没有丢弃失败记录。
- [合并前本地工作历史](focus-stack-local-work-history.md)：旧阶段编号与实验流水。
- [连续画作参考验收历史](focus-stack-reference-validation.md)：旧10/84张、tone/owner研究及拒绝结果。
- [张好好历史诊断](focus-stack-zhang-haohao-diagnostic-2026-09-14.md)：旧93张等实验，不作为当前全量验收。
- [benchmarks目录](benchmarks/)：各版本机器摘要，按总览验收表选择；旧文件不删除、不改判。

原[`.kiro`任务入口](../.kiro/specs/layered-camera-group-focus-stitching/tasks.md)仅跳转到本文件，避免两份任务状态分叉。保留专项旧路径以兼容已有引用，不批量搬动源码和报告链接。

<a id="maintenance"></a>

## 9. 后续维护方式
- 任务的勾选、追加和顺序只在§2列出的对应任务文件里改，不在本文件或专项文档里另设任务表。书画堆栈只有L0（集成/守门）勾选tasks.md。
- 本文件§1在有新验收证据时更新，§7追加最新真实结果；新专项文档加入§8导航。
- 细节、逐次实验、参数与命令放专项文档或机器摘要；历史“待验收/下一步”只代表当时，不反向覆盖当前状态。
- 已通过子集和全量待验收分开；功能实现与画质、性能、桌面、发布验收分开。没有新证据时不自动勾选。
- 需求/设计继续维护技术合同；调整合同必须说明理由和验证，文档整理不改变门禁。
