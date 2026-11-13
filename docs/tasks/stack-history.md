# 堆栈任务历史记录（原PF与N3–N9）

> [返回总入口](../project-progress.md) · 书画堆栈开发见 [tasks.md](../../.kiro/specs/layered-camera-group-focus-stitching/tasks.md)

2026-11-12从`docs/project-progress.md` §3原样迁出（链接已按新位置改写），保留每个条目的完整说明和历次验收记录。**本文件不再勾选**：未完成条目的当前状态和后续安排在[tasks.md §5 储备任务](../../.kiro/specs/layered-camera-group-focus-stitching/tasks.md)和对应任务中维护。

<a id="stack"></a>

### 3.1 性能、观测与基线（PF）

- [x] **PF.1** P0：记录配置/实际路径、fallback、singleton station和阶段资源统计；修复singleton-dominated scan误走高成本路径，保留兰亭行为保护。
- [x] **PF.2** 统一SourcePreflight，在单次运行复用stat、hash、EXIF和已加载来源，报告实际RAW解码次数。
- [x] **PF.3** 有界source/ROI/native fusion缓存与render→QG来源复用；缓存携带owner/source证据及失效键，不只复用RGB。
- [x] **PF.4** 修复默认512MiB/1GiB预算下大对象无法复用：有界匿名无损backing为8GiB/4GiB；同9张冷/热RAW18、fusion6、source/native hits113/15。
- [x] **PF.5** station owner evidence、criterion-specific ROI、独立closure及输出采样密度修复；v28验收、基线固定、v31/v34/v35同9张质量与缓存回归完成。亚像素/多区域对比未证明替代owner稳定胜出，保留原选择规则。
- [x] **PF.6** v34/v35冻结二进制、同9张默认预算ABBA＋BAAB八轮交错归因完成。原始v35冷/热532.148/677.405秒及相对v34的32.51%/80.05%慢跑保留。新八轮v34/v35各4次，wall均值408.20/397.54秒、范围350–471/343–497秒；进程CPU均值1881.59/1888.64秒，融合均值257.14/252.53秒，source＋native后端IO均值40.17/38.13秒，所有轮次QG与v28 TIFF一致。未复现稳定版本级回退；同版本wall大幅波动与并行度/机器负载变化一致，但1分钟负载含任务自身，不能单独证明因果或据此声称v35提速。保留跨素材性能和稳定空闲环境验证边界。[八轮逐次证据](../benchmarks/pf6-v34-v35-interleaved-attribution.json)。
- [ ] **PF.7** 代理图与feature缓存贯穿准备/注册，明确跨运行持久化、失效和尺寸合同；运行内source缓存完成不等于该项全完成。
- [ ] **PF.8** coarse retrieval→top-K→完整匹配及bounded rescue，统计召回、pair数、inlier和成本；现有有限候选生成不等于新架构已验收。2026-11-11增加子项：借鉴ASHLAR置换检验，用确定不相邻的机位对估计数据集内“随机相似度”高分位，先作为Stack_Report附加证据与现有固定门槛对照，不替换门槛（见[tasks.md任务7](../../.kiro/specs/layered-camera-group-focus-stitching/tasks.md)）。
- [ ] **PF.9** profile驱动的图割/分块复用、嵌套并发和解码预算优化；先证明瓶颈，再改默认行为。
- [ ] **PF.10** 可替换注册器、模式分类和经典融合对照：FAST/BRIEF与ORB/AKAZE/ECC等、owner/graph-cut与多频带/小波等，测真实质量与资源后再决定默认路线。2026-11-11按书画定位调整：注册部分（含LightGlue+DISK/ALIKED、RoMa边界rescue，不用SuperPoint/MASt3R）保留并前移；融合部分只保留OwnerHybrid，PMax/小波等通用基线降为离线参考。
- [ ] **PF.11** scene-linear边界、持久化Virtual_Tile/按需tile、可选GPU及learned fusion研究；保留CPU fallback、来源和QG证据，研究结果不得直接替换生产路径。2026-11-11：按需tile仅在S2证明全量超内存时提前；GPU与learned fusion移出当前顺序。

技术方案、开源调研和历次数据分别见[性能方案与记录](../focus-stack-performance-optimization.md)、[开源算法对比](../focus-stack-open-source-comparison.md)。其中历史P0–P4及阶段0–8不是第二套当前待办，以PF条目为准。

<a id="n3"></a>

### 3.5 N3：观测与真实素材入口

- [x] **N3.1** 11套数据manifest、输入/参考排除规则和regression runner接线；马远14个PSD作为14张合成源图，未展开内部图层。
- [x] **N3.2** v35两套10张及9张正对照已绑定构建、输入、运行参数、network/资源、报告和输出；运行完成与验收通过分开记录。
- [ ] **N3.3** 全路径统一Stack_Report及最终画布诊断，保留source/native/world变换、候选生命周期、拒绝阶段和provenance；当前已覆盖的路径不代表legacy/全部素材齐备。
- [ ] **N3.4** 每次行为变化先做数据契约，再做受影响真实回归；涉及兰亭成片时确认45/45、整图/关键crop、参考digest、Coverage/Ownership、`output.written`与文件一致。
- [ ] **N3.5** 完成10张稳定复验后，另行安排阆苑84/文苑126及其他素材的解码、机位图、资源、停止原因和实际输出验证；不复用旧报告冒充新验收。2026-11-11起按[tasks.md任务5](../../.kiro/specs/layered-camera-group-focus-stitching/tasks.md)执行：先全量D1摸底并产出诊断卡，再按区域组逐级推进。

<a id="n4"></a>

### 3.6 N4：跨机位几何与闭环

- [x] **N4.1** v32/v33在当前两套10张中修正station-local/global seed混用，核对原生倍率、anchor、crop、像素中心和SourceSamplingMap；保存候选失败链。
- [x] **N4.2** v34有界倍率证据契约：候选与最终solver共用独立留出、多区域和Jacobian检查，几何倍率与画质采样分开；Source_RAW和原质量阈值保留。
- [x] **N4.3** v35文苑先联合优化再剪边，分离几何修正与输出密度；真实10张保留3条关系，原生P95=1.810738px、输出P95=2.136105px，closure converged、完整原生验收通过。
- [x] **N4.4** 阆苑10张独立闭环：0–2已由原生双向匹配与独立留出支持，三条关系参与有界联合场；输出像素闭环、原有Quality Gate、原生TIFF及同二进制重复运行通过。此项只收口真实10张子集，不代替阆苑84张或其他素材验收。

  v36已实现有界几何重叠域光度补采样，原成功测量不变，保留120样本/4px Coverage及全部质量阈值。首次完整测试暴露稀疏遮罩拒绝回归，限定仅几何窄重叠补测后，完整测试1036通过/0失败/41忽略。真实10张0–2光度差0.012708、边缘比0.828772、方向差5.0593°，三项缺失拒绝消失；QG pass/8ROI、TIFF与v35相同，但10个留出点、四区2/5/3/0及1.0183%尺度重拟合差仍未通过，closure=no_constraints、完整验收拒绝。wall505.728s、RSS9,396,682,752B。下一步补足独立几何采样，不更改门槛；本轮未重跑9张候选回归，旧基线保留。[v36复验摘要](../benchmarks/langyuan-v36-overlap-acceptance.json)。
  v37增加窄重叠的独立原生候选网格，保留原Coverage/双向匹配/拟合/留出门槛。定向及完整库测试通过（1037/0/41），阆苑10张实跑仍拒绝：最终36点、留出10点、四区2/5/3/0、尺度重拟合差1.0183%不变，新增候选未通过，QG pass/8ROI但closure=no_constraints。此实现尚未解决N4.4，不能称闭环修复完成。下一步追踪探针触发、Coverage/双向匹配淘汰及候选选择，区分未采到与测量失败。9张冷/热回归均通过，TIFF与v28一致、RAW18/fusion6、source/native hits113/15，wall413.441/400.706s、RSS10,037,919,744/9,583,984,640B；现有基线保留。[v37复验摘要](../benchmarks/langyuan-v37-native-probes-acceptance.json)。
  v38（2026-11-08）已修复补采样触发/去重缺陷：原按粗层候选数判断不足，且失败候选挡住附近新探针；现按原生验证后的有效点数触发，仅与有效点去重。实跑0–2某候选187个粗层点仅36个通过；新增1061探针中源Coverage拒绝191、目标Coverage拒绝439、双向匹配拒绝407、测后Coverage拒绝1、接受23。最终选中56点/16留出、四区2/7/6/1，尺度重拟合差2.9652%、部分区域误差超限；所有备选均未过原门槛，并非合格候选被选错。QG pass/8ROI，closure=no_constraints，完整验收仍拒绝。原生网格/光度定向测试及完整1037/0/41测试通过；9张冷/热全部基线检查通过、TIFF保持一致、RAW18/fusion6、hits113/15，wall364.964/567.781s、RSS12,493,488,128/9,299,951,616B。剩余：拆解双向匹配内部拒绝与窄域模型条件性，补足独立空间证据；本项继续未完成。[v38分阶段证据与验收摘要](../benchmarks/langyuan-v38-probe-audit-acceptance.json)。
  v39–v41（2026-11-08）已细分原生双向匹配淘汰原因。0–2的1061个独立探针中，源/目标Coverage分别拒绝191/439，前向相关性拒绝399，前向边界拒绝6，反向相关性拒绝2，最终接受23；主要瓶颈不在往返误差。56点实际四区为13/20/8/15，但整图棋盘留出仅2/7/6/1，且点云主轴方差比约326，投影拟合在窄带外病态。v40按真实重叠点分层留出后，选中候选达到25拟合/31留出、四区7/10/6/8；第三区留出误差5.251px、全图尺度重拟合差2.171%，仍超过3px/1%原门槛。v41加入同一原生点集上的有界仿射候选，经原完整门禁评估：其尺度差0.0096%，但第三区误差16.046px；投影与仿射均未通过，0–2不接边、closure=no_constraints、阆苑10张完整验收仍拒绝，QG pass/8ROI。v41定向及完整测试通过；9张NEF冷/热全部基线检查通过、TIFF与v28一致、RAW18/fusion6，wall334.946/334.768s、RSS11,966,660,608/12,460,277,760B。期间脚本缺包装器、旧二进制路径和执行权限的无效运行已留档并排除。下一步只针对左端真实窄重叠补足可靠独立匹配，验证四区误差及1%重拟合；不得凭候选数量或闭环路径组合放行。[v41冻结证据与验收摘要](../benchmarks/langyuan-v41-affine-hypothesis-acceptance.json)。
  v42–v43（2026-11-08）对31px前向相关性失败点增加23/15px双补丁独立双向复测，要求两次落点在1原生像素内一致，原相关性/峰值/往返门槛不变。v42未限定范围，导致9张NEF输出摘要及Coverage审计回归，已拒绝；v43仅在窄预测重叠且倍率越出近1:1范围时启用。阆苑0–2的独立网格新增4个局部一致点，选中候选64点、31拟合/33留出、四区8/11/6/8，四区误差1.234/0.455/2.183/1.427px，尺度重拟合差0.1223%，均通过原几何证据门槛。三边闭环进入有界station field，但输出密度1.247070后直接配对P95=3.150047输出像素，超过原3px门槛；故阆苑完整验收仍拒绝，QG=not_run/0ROI，未写合格TIFF，N4.4仍未完成。下一步改进三边联合几何，使1–2留出P95在输出像素下通过，不能降低输出密度或放宽闭环门槛。v43定向与完整测试1040/0/41通过；9张NEF冷/热全部基线检查通过，TIFF与v28一致、RAW18/fusion6，wall351.577/340.272s、RSS11,703,222,272/11,841,142,784B。[v43冻结证据与验收摘要](../benchmarks/langyuan-v43-scoped-local-consensus-acceptance.json)。
  v44（2026-11-09）用保存的真实三边原生点回放联合场，确认原平滑下1–2独立留出P95=2.526原生像素；仅在跨倍率且重叠短边小于一个64px场单元时使用更强有界平滑，其他路径保留原参数。真实阆苑10张两次独立进程均完整通过：0–2的64点/33留出、四区误差及0.1223%尺度重拟合继续通过，三边closure=converged，最高独立留出P95=2.395931原生像素、在1.247070输出密度下为2.987895输出像素（原门槛3px）；QG pass/8ROI、station owner证据8ROI、16位TIFF写入且两次SHA-256一致。整图及题签、人物关键预览目视未见明显断裂/重复轮廓；余量约0.012px，仍需在扩展素材时重点观察。受影响素材文苑10张同一二进制完整复验通过：closure converged、原生/输出P95=1.810738/2.136105px、QG pass/8ROI、TIFF SHA-256与v35相同。定向、真实点回放和完整测试1040/0/42通过；9张NEF冷/热全部基线检查通过、TIFF与v28一致、RAW18/fusion6，wall379.928/567.105s、RSS11,140,874,240/8,535,441,408B，热跑未显示稳定提速。既有质量门禁、9张基线和未提交修改保留。[v44冻结证据与验收摘要](../benchmarks/langyuan-v44-joint-field-acceptance.json)。
  后续证据复核（2026-11-09）：逐一重新读取阆苑两次和文苑一次的实际TIFF与Stack_Report，三份文件均存在且SHA-256与各自analysis一致；三次均为10输入、QG pass/8ROI、closure converged、16位TIFF artifact_validation=pass、output.written=true、输入与二进制未变化、network=0。阆苑两次输出摘要相同，原生视觉检查记录仍见上述v44摘要。此复核只确认两套10张的成功链；N3.4的兰亭契约和N4.5的全部失败路径仍待完成。

- [ ] **N4.5** 跨素材统一proposal→measurement→shared gate→accepted/rejected，补齐no overlap/support/bidirectional/grid/scale/hull/orientation/closure失败链及线程/候选顺序稳定性。两套子集完成的子项不等于全部路径完成。
      2026-11-09先补报告汇总定向测试：无预测重叠、联合覆盖不足、双向匹配不足、网格不足、拟合前凸包不足和拟合方向失败分开计数；反转候选顺序后`unmeasured_count`及`rejected_by_failure_stage`不变，未测候选不虚构共享门禁原因。单项Rust测试通过（1/0/0）。这是报告层覆盖，尚未证明每种真实失败都正确触发、线程数稳定或所有素材接入；N4.5继续未完成。
      同日发现同一v46冻结二进制对同9张NEF两次运行分别产生fusion7/6、不同TIFF；v45隔离运行与v46重跑均复现既有fusion6/TIFF基线，故存在运行时不确定性。native fusion缓存指纹现将站内帧记录按路径规范排序，保留全部证据字段；定向测试证明仅记录到达顺序变化不改变指纹。v47冻结二进制的9张冷/热均通过既有基线检查（QG pass/8ROI、RAW18/fusion6、TIFF摘要一致），阆苑10张、文苑10张也通过并与v44 TIFF摘要一致，输出闭环P95分别为2.987895/2.136105px。完整Rust库测试1043通过/0失败/42忽略，`cargo fmt --check`通过。两轮9张不足以证明所有非确定性已消除，线程/候选顺序矩阵仍待完成。[v47机器摘要](../benchmarks/native-fusion-v47-order-and-legacy-coverage.json)。
- [ ] **N4.6** 每套素材只有通过可信关系与几何门禁后才使用相应station pose、站内变换及residual；证据不足维持安全回退/identity并记录原因，完成跨素材验证。

方案与坐标细节见[跨机位倍率与几何](../focus-stack-variable-magnification.md)。

<a id="n5"></a>

### 3.7 N5：owner与融合来源

- [x] **N5.1** 同9张建立station owner、criterion-specific ROI、源图/最终像素绑定、透明孔洞追踪和受限几何支持回退；局部来源比较及亚像素稳定性检查完成，未盲改owner规则。
- [ ] **N5.2** 跨素材完善当前运行的winner/runner-up、run-length、重复测量方差、ROI和provenance；能从最终world回放到Source_RAW，不能依赖test-only oracle代替正式证据。
- [ ] **N5.3** 泛化核对focus warp、residual、mask、透明区域、单owner写入和所有fallback；owner coverage不能牺牲effective pixel、gradient/noise、boundary、ROI或Source_RAW fidelity。
- [ ] **N5.4** 几何及独立闭环稳定后再评估跨站融合；smoothness、hard terminal、ROI reservation、content takeover等研究不自动进入生产。
- [ ] **N5.5** 来源锁定修图（2026-11-11用户确认立项，未开始）：在堆栈结果上用画笔改写局部owner，只允许选择该位置有有效Coverage、且已通过配准的贡献帧，不提供任意像素绘制、仿制或生成填充；修改以owner覆盖层非破坏保存，可撤销，原自动结果和来源证据保留。导出与Stack_Report如实记录人工owner区域、所选来源及修改前后owner，人工区域不计入自动Quality Gate通过证据。参考ShineStacker多层输出、PlaneFuse/OpenFocus来源锁定修图，见[开源对比§10.6](../focus-stack-open-source-comparison.md)。进入条件：N6.5兰亭收口，且legacy路径具备可回放的原生owner/source质量平面；先出交互与数据合同设计，再实现。

<a id="n6"></a>

### 3.8 N6：seam、tone、合成与大图输出

- [x] **N6.1** 同9张原生RGBA16 TIFF、alpha/ICC及失败传播修复并验收；v35文苑10张完整原生输出通过。此范围不是全套BigTIFF/Pyramid合同。
- [ ] **N6.2** 跨机位seam仅在双方有效Coverage内求解，记录GraphCut/DP/中线fallback、代价、超时与机位对；跨素材验证边界。
- [ ] **N6.3** Tone_Harmonizer的shared-world证据与owner fidelity泛化验证；空间tone研究保持隔离，不以模糊掩盖色差。
- [ ] **N6.4** TileSource/TileCompositor按需全分辨率读取，tile/whole-frame像素、mask、owner、alpha、provenance等价；BigTIFF/Pyramid的tile、overview、位深、ICC、取消、fsync、原子发布和编码失败合同完整验收。
- [x] **N6.5** 当前工作区兰亭45张显式画作取景的Coverage/alpha、来源、原生质量门禁、主体亮度/色调及CLI合同验收完成；保留45/45原图和只读参考，I1合并后按L1重验。

  **最新验收（2026-11-11，v62原生证据 / v63入口集成）：** 最终owner从原生瓦片读取，保留实际源变换、局部位移、降采样中心与渲染tone/owner增益；焦点备选只在真实Capture_Station内比较。108个边缘网格按实际owner补测，118,893个未解析像素归零，45个来源/44个机位有据可查。原噪声失败点同坐标从0.793804恢复到1.000000，现有QualityGate为pass/8ROI，24个噪声样本均通过；MTF和边界按原规则保留内容不适用，阈值文件与相对floor常量均未改。最终31085×11011 RGBA16 TIFF与v60逐字节相同（`230e48bb…f7da0c`），来源covered=owned=421,256,167，默认完整来源域未改。纸面1008点/1937邻域、跨owner阶跃P95=3，与内部相同；整图及左/中/右/owner边界原生对照复看未见突出的矩形色块，亮度/色调接近只读参考但纸色仍略暖，真实反光保留。
  通用runner现按实际publication crop区分取景与完整来源基线，从native证据读取ownership验证；新增1条取景基线，所有历史条目（含9张NEF）完全保留。完整入口返回0、failures=[]、严格兰亭保护通过。v63原生wall449.970s、RSS8,914,206,720B；同eb9f99b7冻结二进制9张NEF冷/热所有原断言通过，TIFF保持v28、RAW18/fusion6、默认512MiB/1GiB，wall369.644/399.795s、RSS10,774,904,832/10,416,144,384B。Rust1062/0/42，9项原生证据定向测试及Node runner10项/guard6项通过。v61错误的跨机位焦点比较已拒绝；v62原生门禁通过但旧未取景CLI基线不兼容的失败记录保留，未改写成成功。不根据单次耗时宣称提速；旧topology/intra-station degraded保留，非所有指标均可测、非全项目发布。其他未提交修改及I1/书画专项安排保留。[完整机器摘要](../benchmarks/lanting-v63-native-quality-acceptance.json) · [采样/光度重放与基线说明](../focus-stack-artwork-illumination.md#旧拼接路径原生质量证据) · [冻结合同](../../tests/acceptance/lanting-paper-baseline.json)。

  以下v30–v60为历史阶段记录，其“未完成”结论不覆盖上述最新范围验收。

      2026-11-09只读合同审计已完成：冻结v30运行的45/45原JPEG与记录SHA-256一致，参考图未变，实际2.6GiB TIFF和Stack_Report均与analysis摘要一致，`output.written=true`、legacy路径选择一致。实际路径是流式shifted mosaic；内部存储来源Coverage/owner，但最终返回RGB，TIFF无alpha样本，Composition的两个0计数是未测而非零覆盖；Quality Gate仍为`insufficient_evidence`/0ROI，不能凭文件存在记为质量通过。绑定同一TIFF摘要的既有视觉复核为`not_visually_accepted`：概览存在阶梯黑边、周边背景和下部明显色调/画框断续，本轮复看概览同样可见。下一步将流式来源覆盖与owner状态带到报告/输出合同，再修复画布、接缝、边界、色调，并做整图及关键crop实图验收。[只读审计](../benchmarks/lanting-legacy-v30-output-contract-audit.json)。
      随后把流式mosaic最终mask/owner扫描结果单独写入Composition来源证据，保留历史`opaque_pixels`/`union_projected_pixels`语义。v45真实45张测得source covered=421,256,167、owned=420,327,210、canvas=468,357,714像素，发现首写低权重像素有928,957个owner缺口；修复首写owner记录并通过定向测试后，v46同45张covered=owned=421,256,167，45/45输入及参考摘要不变，`output.written`与实际TIFF一致。v45→v46 TIFF摘要变化，owner参与后续低频亮度校正，不能声称纯报告变更；v30→v45已有几何差异，也不能归因于本次观测。v47同冻结二进制再跑45张，来源covered=owned=421,256,167、canvas=468,357,714，TIFF与v46摘要完全相同，输入/参考不变；回归命令因QG`insufficient_evidence`返回非零，属来源观测通过而非质量通过。输出仍为RGB无alpha，整图仍有阶梯黑边、背景及下部色调/画框断续，因此N6.5继续未完成。[v47机器摘要](../benchmarks/native-fusion-v47-order-and-legacy-coverage.json)。
      v48把裁切后的真实Coverage_Mask传给既有TIFF输出编码器；定向测试核对非矩形裁切。兰亭45张冻结实跑的RGBA16 TIFF保留相同33814×13851画布，ExtraSamples为未关联alpha；独立libtiff逐行审计有47,101,547个alpha=0、421,256,167个alpha=65535、0个其他alpha，精确匹配来源覆盖计数；剥离alpha后的RGB SHA-256与v47逐像素相同。峰值RSS为11,443,290,112B，低于原25,769,803,776B门槛。同冻结二进制的9张NEF冷/热全部既有断言通过，TIFF与v28一致，RAW18/fusion6，wall333.532/344.734s、RSS11,398,004,736/11,665,588,224B。完整Rust库测试1043通过/0失败/42忽略，格式检查通过。白底整图复看仍见阶梯状实际覆盖轮廓、上部曝光块、下部拼接/色调断续；兰亭QG仍`insufficient_evidence`，回归命令非零，故仅alpha合同阶段通过，N6.5继续未完成。[v48机器摘要](../benchmarks/lanting-v48-alpha-contract.json)。
      v49只读诊断（2026-11-10）：同一v48冻结二进制与45原JPEG在scale=0.25顺序运行默认、关闭逐图flat tone、关闭整图低频平滑、关闭owner校正四组；四组QG均`insufficient_evidence`。固定顶部背景带的最大20px亮度跳变分别为0.00429/0.00451/0.03739/0.00476；三段tone均在减轻而非制造该位置的曝光跳变，关闭整图平滑尤其恶化。阶梯边界四组均保留，源于非矩形来源覆盖。默认alpha预览的最大完整覆盖矩形`(158,53)–(2040,720)`（2048×839格）宽高比2.8216；原生TIFF alpha逐行重算为`(2608,874)`起`31085×11011`，只读参考图为`31065×11007`，尺寸差20×4像素，诊断裁切的构图也相似。但直接采用矩形会排除78,979,232个已覆盖外围像素（约18.75%），不能凭尺寸接近覆盖默认完整来源输出。下一步受限验证显式画作取景候选，同时保留默认来源域；下部接缝与QG证据仍待解决。[v49 A/B与画布诊断](../benchmarks/lanting-v49-tone-and-framing-audit.json)。
      v50受限实现：只在FlatArtwork流式路径且显式设置`RAW_EDITOR_ARTWORK_FRAMED_OUTPUT`时，从最终来源alpha求最大完整覆盖矩形；要求至少保留75%来源覆盖与65%来源画布，否则失败，不静默退化。兰亭45张1:1实跑得到`31085×11011`画布、`(2608,874)`裁切和78,979,232已覆盖像素排除量，报告同时保留完整来源421,256,167 covered/owned与原画布面积；独立libtiff逐像素核对v48对应区域与v50，RGB/alpha差异均为0，取景内透明像素0。峰值RSS约9.81GB，16位TIFF/ICC/alpha验证通过，参考输入摘要不变。默认未启用路径同冻结二进制复验45张，TIFF与v48摘要完全相同且无publication crop。同冻结二进制9张NEF冷/热既有断言全部通过，TIFF与v28一致、QG pass/8ROI、RAW18/fusion6，wall531.761/481.363s、RSS11,753,996,288/9,392,177,152B；完整Rust库测试1044通过/0失败/42忽略。兰亭QG仍`insufficient_evidence`，取景整图和原生上/下裁片复看仍见顶部亮度变化及底边局部接缝；可选候选不等于默认验收。[v50机器摘要](../benchmarks/lanting-v50-framed-candidate.json)。
      v51来源定位（2026-11-10）：增加仅显式诊断目录才运行的2048长边owner/coverage导出；同45张1:1取景复验的TIFF与v50摘要完全相同，45/45输入和参考只读摘要不变。顶部亮度边界位于owner22（`DSC08875.jpg`）→owner12（`DSC08865.jpg`），两侧真实覆盖；该平坦背景边界亮度差中位数默认约0.0042，关闭整图平滑约0.041，不能把整图平滑当作缺陷根因。底边owner26→17邻接画框/反光，owner校正开启时高分位边界差更大，但跨边内容不同，尚不足以直接判定算法错误或启用按位置淡出。完整Rust库测试1044通过/0失败/42忽略。下一步在来源有效重叠与画作/画框分区内测量同内容差异，再对局部增益设置证据门；不更改当前质量门槛或默认输出。[v51来源边界证据](../benchmarks/lanting-v51-owner-boundary-diagnosis.json)。
      v52同内容重叠测量（2026-11-10）：仅显式诊断目录下，在现有流式分析图中按来源owner采样有效重叠，5×5局部亮度纹理相关系数≥0.8且非饱和后才记录同内容亮度差；默认渲染不使用这些数值。真实45张1:1取景复验仍为45/45、原TIFF摘要及来源覆盖/owner/alpha合同不变，参考摘要不变，4个定向测试与`cargo check --lib`通过，Quality Gate仍`insufficient_evidence`。顶部base12→candidate22有6737个重叠、6637个纹理候选、760个同内容样本，平均有符号亮度差−0.00116、平均绝对差0.00995，但同内容仅分布在候选局部四区中的两区；底边base17→candidate26有1479个重叠、515个纹理候选、0个同内容样本。故目前没有证据支持对底边施加局部增益，N6.5继续未完成。下一步针对画框/反光区建立独立对应或明确不可测区域，补齐legacy质量证据后再评估局部校正；保留既有质量门禁和默认输出。[v52来源重叠证据](../benchmarks/lanting-v52-source-overlap-evidence.json)。
      用户确认`/Users/dp/Downloads/书画拼图/兰亭/DSC08854_FocusStack.jpg`是基本符合预期的视觉参考。其31065×11007画面与v50/v51受限取景的31085×11011画面构图接近，挂杆、画作和底框均完整。后续固定512×182灰度概览平移搜索的最佳位移为(0,0)；当前墙面/挂杆均值分别约74/104，对应参考约105/150（8-bit），画作均值约72/74，差异主要集中在画外背景和挂杆。参考自身也有挂杆亮度切换，不能仅凭概览称当前接缝是优化回退；v30已有上方黑色阶梯边及下方背景杂块。此图现在是N6.5视觉目标，保持只读；下一步按原生裁片局部配准，分清参考的色调处理与代码回退，再修改拼接或色调逻辑。不能仅以画布尺寸或TIFF合同代替视觉验收。[参考基线与对照](../benchmarks/lanting-reference-visual-baseline.json)。
      v53原生owner校正A/B：复用v52冻结二进制，只关闭`RAW_EDITOR_SKIP_SOURCE_OWNER_CORRECTION`对应的owner校正，同45张JPEG以scale=1和受限取景复验；4个定向测试及输入/参考/二进制/源码、Coverage/owner/alpha、31085×11011画布合同全部通过。耗时372.654s、峰值RSS9,774,415,872B，QG仍`insufficient_evidence`。相对参考的512×182灰度概览MAE，墙面30.63→24.48、挂杆46.49→40.63、画作9.25→8.16、底框14.74→11.17；映射同一owner边界的跨边差中位数，底边0.0710→0.0225、墙面0.00396→0.00579，挂杆约0.0134不变。这些是诊断值，不能替代QG；原生顶部裁片还显示参考挂杆纹理与当前来源内容不同，不能把全部差异当作曝光。owner整体中位数求增益的实现文件本轮未修改，最近提交为2026-10-29的`7ec61a12`；但来源/owner变化仍可能改变它的效果。下一步实现同内容重叠支持的受限owner增益与证据不足回退，验证顶部过渡和底边改善能否同时保留。生产渲染代码本阶段未修改，9张NEF基线保留，N6.5继续未完成。[v53原生对照](../benchmarks/lanting-v53-native-owner-ab.json)。
      v54受限owner增益阶段完成：FlatArtwork流式路径改用已应用write_tone后的5×5同内容纹理残差估计，要求相关系数≥0.9、空间分布、稳健离差及不同base owner一致性；增益限制为1/1.08～1.08，在512像素证据格附近衰减，无证据分析格使用单位增益，保留逐图tone及整图平滑。45张原生受限取景复验保留31085×11011、421,256,167 covered/owned、alpha、45/45输入、参考只读摘要和全部文件合同；31,095个样本、7,417候选对格形成358个通过owner格，容量丢弃0。相对v51，参考四区MAE均下降（墙面30.63→24.48、挂杆46.49→40.63、画作9.25→8.16、底框14.74→11.17）；底边跨边差中位数0.0710→0.0225，顶部墙面0.00396→0.00579、增量小于预先记录的1/255区域诊断容差，挂杆0.01345→0.01334。新结果与关闭owner校正组接近，主要收益来自拒绝无同内容依据的owner中位数增益，不能把它归功于强残差补偿。最终16位TIFF的顶部、底框及1024像素原生1:1裁片复看未见新增光晕/结构断裂，底边偏亮带减轻；反光/来源内容差异仍在，兰亭QG仍`insufficient_evidence`，不记为完整画质通过。实跑wall439.279s、RSS9,662,398,464B，单次数据不证明提速。同冻结二进制9张NEF默认512MiB/1GiB缓存冷/热全部既有断言通过，QG pass/8ROI、TIFF与v28一致、RAW18/fusion6，wall396.937/390.787s、RSS9,212,133,376/10,382,491,648B。完整Rust库测试1047通过/0失败/42忽略，格式及diff检查通过，未提交修改保留。默认完整来源画布规则未改，真实视觉验收使用既有显式取景开关。下一步补齐legacy QG可测证据，并按参考逐区处理仍存在的反光/来源差异。[实现与验收方案](../focus-stack-owner-gain.md) · [v54机器摘要](../benchmarks/lanting-v54-evidence-constrained-owner-gain.json)。

  用户随后明确否决v54整图观感（整图亮度、色调、拼接块明显），上述局部指标不能作为整图视觉通过。v55在同一v54冻结二进制上只关闭整图平滑，墙面/挂杆/画作/底框参考灰度MAE降到7.19/12.11/4.61/6.28，确认原宽范围RGB均值把不同材质混为曝光依据，导致整片压暗；该对照期间工作区正在开发v56，故v55的workspace source_unchanged为false，冻结二进制、输入及参考摘要均未变，不能称为当时工作区源码的新构建。
  v56修复与复验：仅FlatArtwork流式路径改为亮度相似度约束的宽范围估计，12个重叠亮度区间、600长边分析图、三次盒滤波，标量亮度增益保持RGB比例并保护明亮材质，其他堆栈路径维持原函数。45张scale=1受限取景完整复验的所有来源/文件合同通过，Coverage和owner诊断图与v54摘要完全相同；最终16位TIFF整图、顶部、画作和底框原生裁片复看，v54的整片压暗明显减轻，未见新增笔画光晕。参考四区灰度MAE相对v54为墙面24.48→5.42、挂杆40.63→12.31、画作8.16→4.92、底框11.17→7.63；墙面均值104.94、参考104.68。墙面/挂杆色度误差减小，画作/底框色度误差略增；挂杆跨边差中位数0.01334→0.01820、底边0.02254→0.02794，局部接缝/反光仍未解决，不以整图亮度改善掩盖这些代价。兰亭QG仍insufficient_evidence，N6.5继续未完成。原生wall466.294s、RSS10,121,494,528B，不据单次实跑声称提速。完整库测试1048通过/0失败/42忽略，格式和diff检查通过；同冻结二进制9张NEF默认512MiB/1GiB冷/热全部既有断言通过，QG pass/8ROI、TIFF与v28一致、RAW18/fusion6，wall353.706/436.582s、RSS10,819,649,536/10,868,588,544B。参考只读，未提交修改保留。下一步在已恢复的整体亮度基础上，按实际同材质重叠处理挂杆/底框的局部光度约束，并补齐legacy质量证据。[整图亮度实现说明](../focus-stack-artwork-illumination.md) · [v56机器摘要](../benchmarks/lanting-v56-material-conditioned-illumination.json)。

  2026-11-11用户明确要求“兰亭没有问题，优化才算有意义”，兰亭整图验收作为后续优化能否收口的前提。v57直接更换逐图写入tone后，来源covered增至421,261,628、取景变为31084×11012（x=2610），与冻结合同不符，已拒绝。v58冻结几何后修正RGB，恢复原Coverage/owner/取景合同，库测试1051通过；但v1全边界诊断疑点149→180、底框亮度MAE7.63→8.75、最终16位TIFF任一通道裁切358→6121像素，画质回退，已拒绝，未为拒绝候选追加9张验收或记作有效优化。原报告、输入与参考摘要保留。[两次拒绝记录](../benchmarks/lanting-v57-v58-rejected-tone-candidates.json)。当前在验证独立留出改进约束、未测区域保留原tone、平滑修正场和高亮余量保护；全边界审计新增细匹配，旧/新版分别留档，不以修改审计方法冒充图片改进。N6.5继续未完成。

  同日用户明确“优先消除画作主体色块，真实反光可以保留”。v59留出修正虽通过1053库测试及9张冷/热，但最终兰亭TIFF与v56逐字节相同，未改善最终像素，不记作画质提升。主体纸面独立审计随后证明，整图材质重归一化仍增加纸面残差阶跃；v60默认保留来源重叠tone与受支持的owner残差，关闭FlatArtwork整图重归一化，匹配tone与材质滤波均隔离为显式实验。45张scale=1、31085×11011 RGBA16复验全部来源/输出合同通过，TIFF SHA-256为`230e48bbb964545adfce7bfa60f9a99e33e888d8abe22e0d78684bc458f7da0c`，与v55隔离对照精确相同；参考/原输入未变，原生wall336.090s、RSS9,733,734,400B。纸面1008点、1937组邻域比较，残差阶跃中位数/P95由v56的2/4降至1/3码值；314组跨owner边界P95=3、最大7，与1623组纸面内部一致，唯一超过6码值诊断线的来源边界已做1500×1500原生参考/TIFF裁片复看，未观察到突出的矩形色块。纸面259,777,345像素中通道裁切为0；整体略偏暖与参考纸色/阴影差异仍单列，不声称逐像素完美。
  v60同冻结二进制9张NEF默认512MiB/1GiB冷/热全部基线断言通过，QG pass/8ROI、TIFF与v28一致、RAW18/fusion6，wall526.043/535.064s、RSS9,816,293,376/9,898,377,216B；不据单轮耗时声称提速。Rust库1053通过/0失败/42忽略；新增Node兰亭保护5项测试通过。新增`focus-stack-lanting:guard`冻结TIFF、输入manifest、参考、来源Coverage/owner、画布、alpha/位深和纸面回归；`--regression-only`实测通过且明确标记quality pending，默认严格模式实测因legacy QG不足而拒绝，不能放行整体优化。真实未完成项为legacy路径的最终原生质量平面未接入QualityGate（quality_planes_unavailable，ROI=0），不是通过改阈值或虚填置信度来解决；下一步保留实际采样变换、owner和胜出/备选证据后接线现有门禁。当前状态为主体色块回归完成、兰亭整体质量仍待验收，N6.5不勾选。[v60机器摘要](../benchmarks/lanting-v60-paper-tone-baseline.json) · [冻结合同](../../tests/acceptance/lanting-paper-baseline.json) · [实现与保护入口](../focus-stack-artwork-illumination.md)。

<a id="n7"></a>

### 3.9 N7：Quality Gate与最终发布

- [x] **N7.1** v35精确物理来源域覆盖：分母独立于alpha/owner/亮度，域外alpha不能补偿域内孔洞，98%阈值不变；保留历史Composition计数及新审计。文苑/阆苑QG均pass，但只有文苑完整原生验收通过。
- [x] **N7.2** 阆苑受限PS参考注册：原2048参考格3px/40-inlier门槛、独立留出与多区域支持，仅在验证凸包内做辅助比较，不把PS像素/变换用于生产。
- [ ] **N7.3** 所有路径的几何/closure/owner/tone/boundary/证据不足失败都准确反映在最终发布状态；QG、`output.written`、报告与文件系统一致，preview/not_run不冒充最终通过。
- [ ] **N7.4** 阆苑10、文苑10及后续84张满足连续可复现的几何、融合、owner、tone、boundary、输出和资源门后，才评估全面blocking mode；当前验收harness已有拒绝规则继续保留。
- [ ] **N7.5** 书画素材交付等级（2026-11-11新增）：D1能出预览→D2原生出图且来源合同通过→D3 QG通过→D4用户视觉确认→D5两次独立运行摘要一致、峰值RSS≤24GiB且回归保护冻结；只有D5算该素材交付，判据见[tasks.md §1](../../.kiro/specs/layered-camera-group-focus-stitching/tasks.md)。当前：阆苑/文苑前10张D3，睡觉9张子集D5，兰亭D2，其余未达D1或待合并后复验。

<a id="n8"></a>

### 3.10 N8：确定性、资源与provider

- [x] **N8.1** 同9张冷/热输出与归一化业务报告一致；默认缓存/backing预算、解码/融合次数及RSS有观测和范围化检查。
- [ ] **N8.2** 1/2/8/16线程、重复运行、改名/乱序、取消、内存超限、tile/whole-frame及provider fallback矩阵；当前两轮冷/热不能替代该矩阵。I1f的9张热跑与枫鹰（约14GiB）并发时0→2/1→2站点关系内点不同而失败，I1g独立热跑通过，负载相关的不确定性须纳入此矩阵。
- [ ] **N8.3** 统一Rayon/OpenCV预算、缓存key、显式拷贝、RSS采样与临时文件生命周期；检查排序、浮点归约、RANSAC seed和失败原因确定性。
- [ ] **N8.4** Rust默认provider保持；Python/OpenCV只在固定proxy fixture做匹配、重投影、seam/tone/owner、RSS、耗时与digest对照，未证明收益不切默认。
- [ ] **N8.5** 多agent并行基础设施（2026-11-11新增，2026-11-12按用户决定修订）：两个资源槽，`build`（编译/单测）与`run`（真实素材/回归矩阵）各1个，复杂任务排队，CPU与内存设上限不占满机器；所有worktree共用`src-tauri/target`，编译后冻结到`output/acceptance/binaries/<label>/`；输出/摘要/版本号隔离；lane按代码区域分组，新素材接入后归组而不新增lane；同时等用户看的素材≤3套、按批汇总；L0唯一合入与回归矩阵（快速/中等/扩展/全量，中等层有界）。任务1–4见[tasks.md第0批](../../.kiro/specs/layered-camera-group-focus-stitching/tasks.md)，规则见其§3–§4。

<a id="n9"></a>

### 3.11 N9：后续算法研究

- [ ] **N9.1** 现有真实门禁稳定后再研究FlatArtwork similarity、轻微曲率分段homography、受约束非刚性、裁切/透明/边缘填充；v34倍率契约不代表所有模型已实现。2026-11-11升为书画正式任务（S4/L11），仅在S2诊断证明有素材因纸面起伏或装裱变形失败时启动，先test-only研究。
- [ ] **N9.2** 每个新模型保留独立baseline、holdout、失败回退、资源预算和开关；与PF.10/PF.11共用对照结果，避免重复建任务。
- [x] **N9.3（职责迁移部分）** `processing.rs` facade及采样/滤波/颜色/几何helper迁移已由N1-10完成；未来行为变化按N1.12重新验证，旧任务表中的重复“待拆分”不再另开任务。
