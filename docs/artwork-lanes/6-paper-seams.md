# 任务6：兰亭纸面接缝

- 起点：v1 `53348b36`；工作区`.worktrees/task-6`，分支`lane/task-6/paper-seams`；同一执行者与L0，不启动并行agent。
- 状态：实现、受影响素材、快速层及中等层全部通过，L0接受该任务范围并本地集成；用户复看待确认，仅D3，不计D4/D5。
- 目标：兰亭45张、显式取景、scale=1；纸面跨owner阶跃P95≤3，>6处数不超过同工具基线，QG pass/8ROI；输入/参考/owner/coverage/canvas不退化，RSS≤24GiB；视觉变化进入复看队列。
- 保护：原始TIFF SHA `22b5a3717a2411f728b84d0ec1e4a1f852485abd38de43fc027610f8fedf35ac`；原始合同与其摘要保存在starting-state.json，不覆盖主工作区基线。
- 修改区域：流式mosaic的owner_overlap_gain.rs（同位重叠残差联合校正）、必要的独立验证/诊断；不修改共享QualityGate阈值、全局开关、源图、其他素材基线。未改全局热点stitch_images.rs/report_schema。
- 方法：原有增益为fallback；候选只使用实际有效重叠、匹配纹理及分布支持，按相互独立的patch划分fit/holdout，联合拟合来源差异；保留8%界限，并按holdout相对既有增益的改善选择应用。
- 资源：编译/测试占build（jobs≤3），冻结候选；真实验证占run（Rayon≤5、nice≥10）。输出在`/Users/dp/Desktop/raw-editor/output/acceptance/task-6/`，每次运行新目录。
- 像素绑定：旧严格保护锁定TIFF字节，像素修复将保留旧合同的预期SHA差异。只有机器目标通过后才在隔离分支准备候选合同/本素材预期；质量阈值不放宽。按tasks.md §6.3及复看队列规则，提供具体候选、原生裁片与差异后请求用户复看，不自授D4。

## 记录

初始记录：历史v57几何反馈及v58视觉/边界退化已读取，不采用参考图拟合生产颜色或几何。首轮未达标，第二轮结果见下。

### 基线与审计环境

同一原始TIFF在正式资源槽环境重新审计：1008纸面点、506跨owner边；P95=4、max=7、>6=2。与任务4结果一致。证据：`output/acceptance/task-6/baseline-audit/{paper-tone-audit.json,verification.log}`。

受限执行环境直接调用sips时，原始参考JPEG SHA仍相同，却生成另一套参考缩略像素；会得到P95=3、max=8、>6=3，不能算接缝改善。三次受限转换一致，但正式槽转换恢复历史字节。对照保存在`output/acceptance/task-6/converter-check/`；新审计记录源图、TIFF、owner、两张缩略图及解码RGB摘要，不把不同转换底板混合比较。

正式参考缩略图SHA：`affaf67b4a64d74b9b0799e02c28114f730d95a434dc06b6876321aec5802683`；受限转换为`396d2129a5243557519d7312d6880689de40df2fc13a1d6e96a68bcb5474a5f4`。新增保护允许合同显式绑定审计底板及>6计数；旧合同兼容，未放宽任何既有检查。在临时加严合同（相同TIFF、P95仍4、>6≤2、绑定底板）下，旧结果strict通过。

### 实现/定向验证

- 原校正只给候选来源设置其相对当前底图的残差，忽略底图owner稍后的残差增益。现在以每个有支持的512px格为图，解`g_candidate-g_base=log(base/candidate)`，以旧场为弱锚（ridge=0.05），64步阻尼Jacobi；对称处理成对来源，保留±ln(1.08)限幅。
- 8px步长、5×5互不重叠patch交替fit/holdout；每侧至少4个patch，fit MAD≤0.025、至少两个空间象限。只用fit解图，用holdout在最终插值场测残差；mean改善至少5%，median/P95不变差才全局启用，否则精确保留旧场。使用旧方法的区域支持半径，不外推到无证据区域。
- 新增Rust测试覆盖传递约束、有限增益、未支持区域、插入顺序确定性，以及holdout冲突精确回退。构建前已有ONNX运行时先核对SHA再复制到忽略目录，无需下载。
- `node --test tests/focus-stack-lanting-guard.test.mjs`：8/8通过，含>6增加/缺失和审计底板伪装的拒绝。
- 初次定向Cargo调用误用工作树独立target，主动INT停止（exit130）；重试转到共享target。首次共享编译发现重复类型别名，修正后重新编译。保留`overlap-unit-r{1,2,3}.log`，不掩盖失败。

### r1：holdout改善但任务指标未达标

- 源码提交`1293c148822d4a9e5035c99d64efdf48db68f085`，冻结`task-6-paper-r1`，binary SHA `18f92268fc821bf0b00e8b53b4de703e12c3db6dc38b82ed3dc25b4a57b83a41`，线程探针verified（1/1）；真实run为Rayon5。
- `matrix.py --tier quick --binary task-6-paper-r1 --lane task-6 --label r1`：7组全通过；Rust1097通过、0失败、44忽略；`matrix-r1-quick/matrix.json`。
- `run.py --binary task-6-paper-r1 --lane task-6 --dataset lanting --framed --label r1`：QG pass/8ROI、unresolved0、native detail_repairs100（旧97）、RSS12,516,835,328B（11.657GiB）通过；与quick相交，contended，非性能结论。
- holdout90948 patch：mean log残差0.015208→0.013917，median0.009368→0.008146，P950.049871→0.048821。插值场独立指标改善，允许候选启用。
- 但纸面跨owner **P95仍4、max7、>6仍2**；>3边由30降到28，仍未达到≤3目标。RGB残差仍6/1/1，画面MAE artwork4.05→4.01，但rail5.53→5.69；部分边界变差。因此不准备新像素合同、不入视觉验收队列、不勾选任务6。
- TIFF SHA `731073bd38c83dc4643a7eefea5103c0ca0269effc788d7fb1c02729d920b0bf`。严格及regression-only保护均只报`output_pixels_changed`；输入/参考/owner/coverage/canvas/来源和原生QG均通过。保留原始rejected，不改写为通过。
- 证据`output/acceptance/task-6/lanting-r1/{lanting/analysis.json,paper-tone-audit.json}`；[机器摘要](../benchmarks/lanting-task-6-paper-seams.json)。同冻结版正在`--exclusive --label r1-exclusive`复跑并导出test-only重叠样本，先排除负载与确定性差异。

### r2：直接拟合实际渲染的增益基函数（算法说明）

保留r1图解为初值；用fit patch位置上的真实9邻格衰减/归一化系数，建立成对稀疏约束，64步有界coordinate descent。每个pair-cell总权重相同，旧场ridge仍0.05；holdout完全不参与拟合。原local_gain函数逐字保留，拒绝时不因浮点运算重排改变旧像素。新增测试核对拟合基与渲染基在稀疏支持和格边界一致、弱支持的残差改善、未约束节点不变、限幅。

r1独占复跑已完成：TIFF与首轮相同（`731073bd…`）；QG pass/8ROI、P95=4、max=7、>6=2，owner/coverage/canvas/输入保护均通过，RSS12,517,588,992B，contended=false。因此保留未达标结论，并非并发负载导致。`r1-overlap-evidence.json`（13.05MiB）含13399对格及211211样本，用于定位真实插值约束，不拟合参考图。

### r2真实结果：纸面目标通过，进入集成回归

- 源码`72af6ea0bf011b3aea1d59a0e4cfe6570ac1e22b`，冻结`task-6-paper-r2`，binary SHA `de905b0e6e1f3f620a88db2449a6df4f1672212d8e8f7fdc71862960f93db522`，线程探针verified；6项定向测试、快速层7组全过，Rust1098通过/0失败/44忽略。
- 完整45张、44站点，scale=1、显式取景；QG pass/8ROI、unresolved0、detail_repairs101；RSS11,103,862,784B（10.34GiB），内存门通过，contended（与quick相交），不作性能结论。
- **纸面跨owner P95 4→3、max7→6、>6处2→0**；1008纸面点、506边。paper interior P95也4→3。RGB残差6/1/1不变；灰度MAE artwork4.05→3.99、bottom5.41→5.23，但wall6.59→6.66、rail5.53→5.57，保留轻微增差。holdout90948：mean0.013670、median0.008013、P950.047602，优于旧场及r1。
- TIFF SHA `29852099097a03e489d5addc0b04fca53860ef82e2a62ca3c08f0cffbda3ff02`；画布31090×11008、owner/coverage SHA与旧合同一致，覆盖/owned421257787、输入/参考和所有原QG判据不变。源像素校正发生在冻结几何后；细节修补的测量随亮度变化，记录由97变101，未改修补阈值。
- r2首次旧v3合同严格保护只报`output_pixels_changed`，原始rejected分析保留。目标通过且整图/全部6对裁片已实际打开复核后，准备本素材v4：固定新TIFF；P95上限4→3，新增>6上限0与固定参考缩略图SHA；owner/coverage/canvas/input绑定逐项不变。旧合同完整保存在priorBaseline/git，不改共享QG阈值、全局默认开关或其他素材基线；`focus-stack-regression-baselines.json`所有旧锐度floor完全保留，不用rebaseline。
- 本素材v4 strict及regression-only通过，单独记录`r2-v4-{strict,regression-only}.json`，不改写首轮分析。数据绑定改动后再次跑四组Node合同/测试，通过。
- 复看包`output/acceptance/task-6/review-r2/README.md`：整图+6对512×512 RGBA16原生裁片，逐张PNG/TIFF解码RGBA一致、原ICC保留；采样坐标及源/裁片摘要见manifest。导出占run槽并监控RSS，峰值4,262,199,296B（3.97GiB）。Agent看图未见这些位置有明显新笔画光晕/断裂或真实反光丢失；此为技术复核，不代替用户D4。已登记[复看队列](review-queue.md) R20261118-1，1/3，待看。
- 文档保留策略：已逐字节核对r1两次TIFF一致，删除较旧重复TIFF，保留较新的r1-exclusive失败TIFF及两轮全部日志/报告；映射在`duplicate-r1-tiff-retention.json`，不触碰旧正式基线。
- 下一步：rebase到v1、同一冻结r2运行中等层5例。未通过前不勾选任务6或宣称合入。

### 中等层进行时的兼容性核对

9张冷跑已0项不符：TIFF仍v28 `718a903d…`、RAW18/fusion6、默认512MiB/1GiB预算、QG和RSS门均通过。热跑及后三例继续。

发现新guard CLI对旧合同也无条件读取新缩略文件，会使旧合法目录缺图时被拒绝。新增合成CLI plumbing fixture实际复现（`legacy-cli-before-fix.log` exit1），然后改为仅在合同声明referencePreviewSha时读取；不改变v4分支的严格判定，不改Rust或任何运行预期。19项Node测试（guard9/runner10）通过，真实r2的v4严格保护再次通过（`r2-v4-strict-compat.json`）。这不重跑或伪造原生图像检查。

### 集成验收（2026-11-18）

- `git rebase v1`确认工作分支基于最新v1，无需改动；`matrix.py --tier medium --binary task-6-paper-r2 --lane task-6 --label r2`：**5/5通过，均首轮0项不符，无独占重试**。
- 9张冷/热均TIFF=v28、RAW18/fusion6；阆苑10、文苑10均闭环converged、QG pass且TIFF保持原冻结值；所有素材输入/参考/来源/二进制/缓存/内存门通过。中等层最大RSS11,516,854,272B（10.726GiB）。
- 兰亭第二次完整45张与首次r2同一二进制、相同TIFF SHA `29852099097a03e489d5addc0b04fca53860ef82e2a62ca3c08f0cffbda3ff02`，纸面审计全字段相同：P95=3、max=6、>6=0；v4 strict与regression-only均pass；owner/coverage/input/canvas保持。重复运行目录`matrix-r2-medium/lanting-framed-attempt-1/lanting`，不改写首次旧v3拒绝。
- 最新Node19项全过；冻结原生719项构建输入与最终工作树完全一致，原生仅改owner_overlap_gain.rs；运行时SHA保持。r1/r2的原始13399对格样本逐项相同，改善不是采样漂移。
- L0结论：任务6规定的实现/机器验证/视觉入队条件已满足；本地合入v1后以同一已验证可执行文件发布`task-6-paper-verified`冻结label（保留72af6ea0原构建提交，另记冻结提交，不伪称重建）。原始v3、r1失败及首次r2像素变化拒绝均保留。未请求push、不打包桌面应用。
- 复看包R20261118-1已实际提交，待看1/3；agent已看整图及6对原生裁片，但用户尚未确认，所以仅D3，不记D4/D5。第1批下一项为任务7离线置换检验；其他全量素材及未取景路径不从本次外推。
