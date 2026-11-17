# 任务6：兰亭纸面接缝

- 起点：v1 `53348b36`；工作区`.worktrees/task-6`，分支`lane/task-6/paper-seams`；同一执行者与L0，不启动并行agent。
- 状态：实现与原生验证中，尚未验收。
- 目标：兰亭45张、显式取景、scale=1；纸面跨owner阶跃P95≤3，>6处数不超过同工具基线，QG pass/8ROI；输入/参考/owner/coverage/canvas不退化，RSS≤24GiB；视觉变化进入复看队列。
- 保护：原始TIFF SHA `22b5a3717a2411f728b84d0ec1e4a1f852485abd38de43fc027610f8fedf35ac`；原始合同与其摘要保存在starting-state.json，不覆盖主工作区基线。
- 修改区域：流式mosaic的owner_overlap_gain.rs（同位重叠残差联合校正）、必要的独立验证/诊断；不修改共享QualityGate阈值、全局开关、源图、其他素材基线。未改全局热点stitch_images.rs/report_schema。
- 方法：原有增益为fallback；候选只使用实际有效重叠、匹配纹理及分布支持，按相互独立的patch划分fit/holdout，联合拟合来源差异；保留8%界限，并按holdout相对既有增益的改善选择应用。
- 资源：编译/测试占build（jobs≤3），冻结候选；真实验证占run（Rayon≤5、nice≥10）。输出在`/Users/dp/Desktop/raw-editor/output/acceptance/task-6/`，每次运行新目录。
- 像素绑定：旧严格保护锁定TIFF字节，像素修复将保留旧合同的预期SHA差异。只有机器目标通过后才在隔离分支准备候选合同/本素材预期；质量阈值不放宽。按tasks.md §6.3及复看队列规则，提供具体候选、原生裁片与差异后请求用户复看，不自授D4。

## 记录

尚未产生修复候选的真实验收结果。历史v57几何反馈及v58视觉/边界退化已读取，不采用参考图拟合生产颜色或几何。

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

### r2：直接拟合实际渲染的增益基函数（待验证）

保留r1图解为初值；用fit patch位置上的真实9邻格衰减/归一化系数，建立成对稀疏约束，64步有界coordinate descent。每个pair-cell总权重相同，旧场ridge仍0.05；holdout完全不参与拟合。原local_gain函数逐字保留，拒绝时不因浮点运算重排改变旧像素。新增测试核对拟合基与渲染基在稀疏支持和格边界一致、弱支持的残差改善、未约束节点不变、限幅。
