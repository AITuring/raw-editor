# 连续画作参考验收记录（2026-09-18）

本文记录一次针对长幅平面画作的参考驱动验收，以及为该类输入保留在生产路径中的修复。
参考文件只用于建立可信几何和诊断对照；生产拼接器不会把参考图像像素复制进结果。

## 本次保留的算法修复

- 自动模式只有在投影模型的内点覆盖了足够大的画面区域时才接受单应性。重复的局部笔画即使
  能拟合出很小的残差，也不会再把整个平面拉成错误的投影。
- 多尺度特征不足时，先对固定大小的局部单元做有限对比度拉伸，再检测 FAST/BRIEF；描述子仍
  从原始分析图提取，因此局部拉伸只帮助发现低对比结构，不会改变匹配像素。
- 流式 mosaic 的色调场只从当前拍摄组已经拥有的重叠估计。拍摄组之间只允许低频、有限幅度
  的增益协调；它不能平均或覆盖源图高频细节。
- 所有权过渡按目标像素双线性采样分析掩膜，而不是给一个分析 cell 写入常量 alpha，避免
  方块边和宽接缝，同时保留被选中的源图细节。
- 最终画布保留全部真实源图覆盖的联合边界，不再用“最大无空洞矩形”裁掉画框。空白角落不以
  拉伸、镜像或生成纹理补齐。

实现位置：

- 自动模型选择和参考验收 harness：[`src-tauri/src/panorama_stitching.rs`](../src-tauri/src/panorama_stitching.rs)
- 流式所有权、色调场和 materialize：[`src-tauri/src/panorama_utils/mosaic.rs`](../src-tauri/src/panorama_utils/mosaic.rs)
- 低纹理局部对比特征：[`src-tauri/src/panorama_utils/processing.rs`](../src-tauri/src/panorama_utils/processing.rs)

## 参考驱动验收链路

测试构建中的 `reference_acceptance` 模块从已处理的 RAW 生成缩略图，然后由三个脚本完成：

1. `scripts/register-stack-reference.py` 用 SIFT/RANSAC 为每张源图估计到可信参考的单应性，并
   要求内点、空间支持、重投影误差和凸四边形检查全部通过。
2. `scripts/refine-stack-reference.py` 用局部对比度图和 DIS 双向光流估计稠密残差。有效区域
   要求前后向一致、位移受限，并按真实网格节点采样，避免通用 resize 带来的半 cell 偏移。
3. `scripts/estimate-reference-tone.py` 只拟合低频 RGB affine 系数
   `reference_low = gain * source_low + offset`，不写入参考像素。Rust harness 可选择逐源或按
   `group_id` 应用这些系数后，再走原有 RAW 加载、局部配准、清晰度评分和 ownership 渲染。

这些脚本需要 Python、NumPy 和 OpenCV；参考清单、生成的 JSON 和大尺寸源图属于本机验收材料，
不应提交到仓库。Rust 测试入口默认被 `#[ignore]` 标记，只在显式提供环境变量时运行。

## Langyuan-10 final-world station seam（2026-10-15，局部改善）

默认 `LayeredVirtualTile` 的 accepted station overlap seam 已修复一个生产接线缺口：投影矩形首尾的
无双覆盖行不再使整条 DP 路径失效；RGB、Coverage 边界距离和 final-world 梯度/笔画代价现在进入同一
确定性最小路径。应用只发生在旧/新硬边界之间且不超过 32px，不混合 owner 或羽化像素。

冻结 `DSC_3680.NEF`–`DSC_3689.NEF` 两跑均只改写 34,846 pixels，preview SHA-256 同为
`bf1ddb06cbc192c281632fa627c86e035c4262ac86af37ec969bb6844b4deb04`。station 1/2 路径上的
stroke-cost P50/P95 从 `0.95498/1.64395` 降至 `0.82819/1.51070`，RGB discontinuity P50/P95
从 `0.11910/0.22777` 降至 `0.08792/0.20772`。owner coverage 仅从 `0.7436796722` 变为
`0.7436481415`；local scale、effective pixel、gradient、noise 和 Delta-E 的 retained first-run
统计与 fresh baseline 相同。10/10 RAW digest 不变，network requests 为 0。

production `boundary_stroke_alignment` 仍只有 2 个 `0.0px` measurement，故继续标为
`not_applicable`，不能宣称完整边界验收通过。Residual_Warp 也继续为 identity：test-only dense
残差 before/after P95 相同，没有满足 round-trip 与严格 post-P95 降低条件。owner `<0.99` 与
effective pixel `<0.98` 仍是已知 blocker。完整变体、rollback、overlays/crops 和验证证据见
[durable report](../.agents/tasks/langyuan-final-world-station-seam-fix-results-2026-10-15.md)。

后续 Option E 调查已把 production boundary sampler 限定到 accepted 跨机位硬接缝，并要求两侧
Coverage、Source_RAW provenance、独立强边缘、方向/极性和空间去重。它不再把一个 P95 汇总值复制成
额外 measurement。fresh Langyuan 因而从两个重复 `0.0px` 暴露出真实可测失败；安全固定截面版本仅有
4 个唯一配对（`1/1/1/2px`），扩窗版本虽达到 15 项以上却同时揭示 station 1/2 多个 `4–27px`
错位，不能通过采样规则隐藏。effective pixel 的缺口精确为净 `1,905,885px`；基于真实 Coverage 的
黑像素恢复可达约 `0.9997`，但使保留 seam 的 station 1/2 RGB P95 从 `0.20772` 回归到
`0.21763`，所以输出行为已撤回，`<0.98` blocker 保持。详见
[后续 A/B 报告](../.agents/tasks/langyuan-option-e-boundary-effective-results-2026-10-15.md)。

2026-10-21 的独立 station-rendering correspondence 调查没有找到可安全应用的几何修正：在 actual
accepted seam `±48px` 邻域中，保持 `0.15` contrast、`10°` orientation/polarity、multiscale patch
唯一性和双向 `<=1px` closure 后，station 1/2 的 229 个 source candidates 只剩 4 个 64px 空间去重
对应；位移为 `[-32,-29] / [-9,-19] / [-19,-4] / [22,-26]`，checkerboard fit/holdout 仅 2/2。
因此 station pose 未改，Residual_Warp 继续 `regions=[] / identity=true`。production sampler 同时从误留的
`±16px` tangent 宽窗恢复为安全 `±2px` fixed cross-section，final fresh 只保留两个 `1px` 测量并诚实
`not_applicable`，不再接纳 4–27px 错边。三次 preview byte-identical（SHA-256
`c25fb9d4…99c6`），但当前树 station 1/2 RGB P95 已为 `0.2176262 > 0.20772`，所以没有重新启用
Coverage 黑像素恢复，也不能宣称 retention gate 或 Option E 通过。详见
[closed correspondence 报告](../.agents/tasks/langyuan-option-e-closed-stroke-correspondence-results-2026-10-21.md)。

## 阆苑女仙回归状态（未通过）

冻结输入为本机验收目录中连续的 84 张 `DSC_3680.NEF`–`DSC_3763.NEF`。几何参考为用户提供的
`未标题-9-yuanshi.jpg`；它的作用是注册和诊断对照，不是生产拼接器的像素来源。

此前参考驱动结果存在错位、分区接缝、明显失焦，并且裁掉了画框，不能作为验收通过或产品输出。
参考图只可作为离线诊断真值，不能参与生产配准、色调或像素合成。

该目录现在是通用算法的回归样例，而不是一次性制图任务。通过条件必须同时满足：自动模式仅输入
原始照片即可完成；完整画框保留；不依赖透视拉正；同一拍摄位置从原始分辨率选择最清晰焦平面；
不同拍摄位置的几何由多焦平面一致证据连接；全图和局部实像检查均无重影、虚焦块和可见接缝。
任何参考匹配合成图都不算通过。

## 批准尺度与 84 张验收（2026-10-08，未通过）

最终 Station Relation 尺度门槛保持批准的闭区间 `[0.95,1.05]`。最终 solver 与
VirtualTile authoritative-cluster 现在共用完整 typed canonical measurements 和同一 rejection gate，
包括 low-frequency mean、edge-strength、edge orientation、方向、逆变换与凸性；prior-repair bound
继续复用同一尺度区间。

评审修复后的最新 run 位于本地 ignore 目录：

```text
.agents/artifacts/approved-scale-and-84-acceptance-2026-10-08/20261008T093406Z/
```

执行命令为 `npm run stack-acceptance:check`（设置冻结 source/report 绝对路径与
`CARGO_NET_OFFLINE=true`）。唯一报告名为
`reports/stack-report-869f27e5-3f4d-4e24-9a85-f646ee013fbf.json`，SHA-256 为
`eaa54e6400c0924553152eaa2f61cde6176639828600ee4250dd05e7e8ffc114`。84 个 SourceRaw
均成功解码且运行前后 digest 不变，network requests 为 0。

实际结果仍为未通过：28 个 station 精确覆盖 84 个唯一 member，无 isolated/undecodable；378 个
关系候选中 8 accepted、370 rejected、316 unmeasured。accepted graph 仍为 20 components，member
counts 为 `[3,3,4,4,3,4,7,6,3,3,7,3,4,6,4,4,4,6,3,3]`，合计 84。失败阶段为
`fitted=32`、`no_predicted_overlap=194`、`insufficient_measurable_patches=95`、
`insufficient_polished_support=22`、`insufficient_bidirectional_matches=20`、
`insufficient_grid_coverage=7`；最终拒绝原因仍是 `station_relation_scale_out_of_range=24` 与
`station_relation_edge_orientation_mismatch=2`。

完整 authoritative gate 后共保留 74 个 prior-free candidate record，其中 31 fitted、22 停在
polished support、10 停在 bidirectional matches、7 停在 grid coverage、4 停在 measurable patches。
规划器尝试修复 station 7、15、16、26、27，但 correction 分别为
`296.4px/0.8270`、`151.3px/0.8997`、`767.2px/0.8270`、`722.4px/0.7989`、
`1315.5px/0.7291`，全部超出既有位移/批准尺度包络，故 applied prior repairs 仍为 0。不能把这些
proposal 直接放行；下一轮连通性 P0 应先持久化并核对这 5 个 proposal 的 tile/native/world 坐标组合，
确认是否存在尺度换算错误，再处理 `no_predicted_overlap`、measurable support 与 polish 丢边。

管线在关系图不连通处停止，因此**未进入最终** fusion、composition、tone、residual warp 和
Quality Gate；station 内部 GraphCut 日志不代表最终融合已开始。测试耗时 1,111.71 秒，peak RSS
10,656,382,976 bytes（2,189 samples），低于 25,769,803,776-byte 门槛。harness 因
`geometry_disconnected` 以 101 退出并写出有效报告。

四个旧 `command.log` / `stack-report-*.json` 已从当前工作区快照删除；新的 raw command log 与报告由
`.gitignore` 保持为本地 artifact，仓库只保留本摘要、关键计数、命令、报告 SHA-256 和 artifact
位置。意外提交 `3771f1ba` 的历史仍含旧 blob；清理已存在历史需要用户另行明确授权，本轮未改写提交。

本轮不放宽尺度、空间支持、方向、重投影、prior-repair 或 Quality Gate 门槛，也没有混入
Langyuan-10/Wenyuan-10 画质修改。

## 方案 E 10 张真实运行（2026-10-12，部分完成）

本轮新增的置信度驱动 Owner 平滑只在 winner 明显优于 runner-up 的 ownership cell 降低正
Potts 平滑；不确定 cell 保留原平滑，并继续执行 disagreement veto、单一 Source_RAW 写入和
Coverage 约束。诊断开关 `RAW_EDITOR_FOCUS_DATA_TERM_ONLY` 仍限制在 `cfg(test)`。

Langyuan-10（`DSC_3680.NEF`–`DSC_3689.NEF`）完成分组、机位关系、Virtual_Tile、fusion、
composition，并生成完整 Quality_Gate 报告。三个 station 的 `owner_sharpness_coverage` 为
`0.999933 / 1.000000 / 0.998985`，但 `boundary_stroke_alignment` 的 P95 约 `6.0px`、最大
`6.06px`，Tone_Harmonizer 边界最大 `Delta_E00=16.94`，报告仍为 `fail/degraded`；
`residual_warp.identity=true` 是因为没有满足启用条件的 16 个双向闭环证据。

Wenyuan-10（`DSC_4829.NEF`–`DSC_4838.NEF`）在 Quality_Gate 前停止：3 个 station 只有
`0→1` 关系通过，另外两条关系的尺度为 `1.1511` 和 `0.8973`，超出批准 `[0.95,1.05]`，
因此形成 2 个连通分量（6+4 张），`quality_gate=not_run`。不能据此宣称梯度、噪声或整组拼接通过。

真实报告、运行命令、输入清单与 SHA-256 manifest 保存在本机
`/private/tmp/raw-editor-e-evidence-2026-10-12/`；运行使用 `CARGO_NET_OFFLINE=true`，
`network_requests=0`。方案 E 保持部分完成；Quality_Gate 的测量仍完整记录，失败结果现在在最终
结果入库和 staged 文件发布边界被拒绝。

## Option E：84 已知限制与 10 张 owner sharpness（2026-10-12）

用户已明确调整优先级：84 张自动连通性作为当前已知限制保留，不能宣称已验收；本阶段不再运行 84、继续 B–F 或放宽任何 Station Relation / Quality Gate，转而调查 Langyuan-10 与 Wenyuan-10 的 owner sharpness。

冻结 10 张夹具是 `阆苑女仙/DSC_3680–3689.NEF` 与 `文苑图/DSC_4829–4838.NEF`。20 个 Source RAW 的运行前后 SHA-256 已复核相同。失败的 B `global_coverage_similarity` 已从 production relation orchestration 断开，仍只保留研究实现与测试。

Langyuan 当前新基线 `owner_sharpness_coverage=0.7277342`，分站为 `0.7795648/0.7107007/0.6970655`。同一配准和评分下的 data-term-only test counterfactual 总体及三站均为 `1.0`，而 shortfall 与 mismatch veto 重合仅 0.43%，证明主短缺来自 graph-cut smoothness/data-cost tradeoff。简单降低 smoothness 或硬钉 5% winner 虽可把 owner coverage 提到 `0.99957–1.0`，却使 512px 单 owner ROI 数降为 0，导致 local-scale、gradient 和 color 证据不足；这些实验已撤回，不能用牺牲其它 Requirement 11 的方式过门。最终保留代码下 Langyuan 的其它上述判据仍 pass，但 owner coverage 仍为 `0.7436797 < 0.99`。

Wenyuan 最终仍在 composition 前被 strict geometry 拒绝：3 stations 只有 `0–1` accepted，`0–2/1–2` 的宽支持模型 scale 为 `0.8938178/0.9055316`，不在批准的 `[0.95,1.05]` 内，形成 2 components，故不能伪造最终 owner/crop 证据。

完整命令、报告路径、输出 hash、分站 owner map、配准变换、counterfactual 和 blocker 见 `.agents/tasks/option-e-owner-sharpness-investigation-2026-10-12.md`。下一步仍限 owner score-noise 诊断；不开始 seam/noise/tone。

## 方案 E 最新检查点（2026-10-21，仍未完成）

本检查点只发布当前真实夹具的可复核状态。84 张自动连通性仍是已知限制，不能把 84 张的
`geometry_disconnected` 改写为通过；两套 10 张的 Quality_Gate 仍不能作为通过证据，20.9 的阻止逻辑
已接入最终结果入库和 staged 文件发布边界，但“两套全部通过后完成 20.9”的前置条件尚未满足。
后续任何共享模块改动都必须在 84 张重新连通后重跑 84 张回归。

Langyuan-10 的固定输入为 `DSC_3680.NEF`–`DSC_3689.NEF`。当前报告为
`/private/tmp/raw-editor-gate/continue-20-8-languan/reports/stack-report-3284d482-37b8-40ca-96b2-1bcff59d57b1.json`，
其 SHA-256 为 `85cadbf8a1476d4351ba400ca6604591b0e729a016530c4057af25d3be179664`；
运行完成且生成了 8762×10944 的 preview-only 预览
`/private/tmp/raw-editor-gate/continue-20-8-languan/langyuan.preview.jpg`；报告的
`output.written=false`，所以不能把这次预览当作最终导出文件。10/10 RAW 均解码，运行前后 SHA-256
一致，`network_requests=0`，峰值 RSS 为 11,509,710,848 bytes（2411 次采样）。

本次 Quality_Gate 有 256 个 ROI，结论为 `fail`：`local_scale_median`、
`local_scale_pixel_ratio`、`gradient_energy_normalized` 和 `roi_delta_e00` 通过，
`mtf50_normalized` 为 `not_applicable`，`noise_sigma_ratio` 的 98 个可测 ROI 通过；
`effective_pixel_count` 有一个 0.9783406 低于 0.98，`boundary_stroke_alignment` 有一个
2.0px 结果高于 1.5px，`owner_sharpness_coverage=0.7436481` 低于 0.99。因此只能记录为
部分完成，不能宣称 Langyuan-10 画质门禁通过。`residual_warp.identity=true`，因为本次没有满足
启用条件的真实残差区域。

本检查点报告是在 effective-pixel 计数接入 `Stack_Report.composition` 之前生成的；当前代码已让
`opaque_pixels` 与 `union_projected_pixels` 复用同一次 Quality_Gate 扫描的精确计数，下一次真实运行
会把这两个字段写入报告。拒绝错误同时保留失败判据及 ROI 世界坐标，方便调用方直接定位失败证据。

Tone_Harmonizer 实际使用 2 条 accepted Station_Relation、3 个 tile，
`pairs_without_evidence=[]`；增益范围为 `0.9809674..1.0118614`，offset 接近 0，三个 tile
均未 clamp。边界检查 44 个窗口，最大 `Delta_E00=16.8033404`，超过 1.5 的窗口记录为
`tone_boundary_delta_e_exceeded`，状态为 `degraded`。该值用于暴露真实内容/照度差异，不能通过
提高阈值、整站截断增益或边界模糊消除。

Wenyuan-10 的固定输入为 `DSC_4829.NEF`–`DSC_4838.NEF`。2026-10-21 当前脏树 fresh strict run
已恢复三机位连通：图像证据分组为 `{4832,4834}`、`{4833,4831,4830,4829}`、
`{4835,4836,4837,4838}`，三条 Station_Relation 全部通过未改变的 `[0.95,1.05]` 门禁。
`0→1/0→2/1→2` 的 scale 分别为 `0.9999504/1.0020960/0.9997123`，hull 为
`0.8595523/0.2899700/0.3660135`，median error 为 `0.6092/1.7078/1.9472px`；三条独立
checkerboard refit/holdout 的 median 为 `0.6148/1.8165/1.8339px`、hull 为
`0.8509/0.2790/0.3397`。报告同时记录不等 tile 尺寸、有效 Coverage bounds、crop translation、
canonical matrix、Local_Scale 与对应样本，排除了 EXIF 方向、pyramid 尺度或 crop 坐标归一化导致
历史 `0.89/0.91` 的解释；历史值来自旧分组暴露的重复纹理错误盆地。

最终报告为
`.agents/artifacts/wenyuan-strict-connectivity-2026-10-21/final-run-2/reports/stack-report-bd1068e5-b780-4946-954e-31c2a3b1d819.json`
（SHA-256 `6e3b20746b3b227ce7bad6eb902a1907508d9fdc99c61882a65bf43c20fec630`）。管线进入
fusion、`8792×8698` composition 和 Quality_Gate；门禁结论为 `fail`，因此
`output.written=false`。fresh 质量目标为 owner sharpness coverage `0.745873 < 0.99`
（station `0.836202/0.671746/0.729700`）和 32 个 gradient ROI（最低 `0.901996 < 0.95`）；
noise 的 76 个可测 ROI 已全部通过。Tone boundary 最大 Delta-E00 为 `18.504616`，closure 因
pair P95 `58.6617px` 回退，Residual_Warp 启用一块区域。本阶段只恢复严格几何并记录这些后续目标，
没有同时修改 owner/gradient/tone/closure 行为。

两次相同 production 参数运行的 TIFF/app JPEG/preview SHA-256 分别稳定为
`a4b50f05…b2deaf`、`beaa008b…3f879`、`42146a3c…1dd0`；10/10 RAW 均 decoded 且运行前后
SHA-256 一致，`network_requests=0`。完整矩阵、对应点、holdout、验证和下一目标见
`.agents/tasks/wenyuan-strict-station-connectivity-2026-10-21.md`。

后续 Option E owner/gradient 调查证明上述 32 项不是实际锐度衰减：旧 gradient ratio 乘以同 ROI 的
`local_scale` 后为 `0.999535..1.007560`。owner Source_RAW reference 已通过完整变换重采样到 output
网格，旧 runner 却只归一化 output、把 reference 当成 scale 1，因而制造约 `1/local_scale` 的失败。
当前修复让两侧使用同一 local scale；两次 fresh Wenyuan 均为 256/256 gradient pass，minimum
`0.993109`，三种输出 hash 与修复前仍完全一致。local scale、effective、noise、Delta-E、boundary、
3/3 strict relation、RAW digest 和 network=0 均无回归。

owner coverage 仍为 overall `0.745873`、station `0.836202/0.671746/0.729700`。逐 cell 的
production/data-term counterfactual 与 best-source gradient oracle 证明 owner 偏软是独立问题；没有把
oracle labeling、ROI reservation、content takeover 或 smoothness/hard-bound 变体接入 production。
Residual_Warp on/off 的 owner reference energy 最大只差 0.56%，且历史失败 25/32 位于 warp 区外。
完整逐 ROI 归因、counterfactual、两跑确定性与 Langyuan no-regression 见
`.agents/tasks/wenyuan-option-e-owner-gradient-results-2026-10-21.md`。

以上报告与本文件只记录结果，不改变 Station_Relation、Tone_Harmonizer、Residual_Warp 或
Quality_Gate 的门槛。对应任务状态和回归命令见
`.kiro/specs/layered-camera-group-focus-stitching/tasks.md` 的 20.6–20.11。当前针对 Tone 的
15 个定向测试、Residual_Warp 的 5 个单测、生产模型构建的 2 个回归、Compositor 的 13 个单测、
Quality_Gate runner 的 15 个单测和残差属性测试的 56 个用例均通过；这些是代码回归证据，不能替代
两套真实夹具的 Quality_Gate 通过结论。

## 2026-10-22 几何连接续做

用户重新授权恢复 84 张机位几何后，生产 prior-free orchestration 接入了已缓存的
`global_planes` 全局覆盖相似度 proposal。它只负责给金字塔匹配提供 bounded-similarity 搜索盆地；
每个候选仍经过双向金字塔匹配、多模型 selector 和完整 Station_Relation 门禁，只有
`selection.passes` 才能写入 `tile_relations` 或 prior repair，`[0.95,1.05]` 及其它阈值没有改变。

真实 pair `1-2` 的报告为
`/private/tmp/raw-editor-global-pair-1-2/1-2/stack-report-22e44f6d-ac6e-4413-ac7e-67cb782bde05.json`：
全局候选实际生成 8 个 proposal，但全部在 `insufficient_bidirectional_matches` 阶段停止，最终
`accepted=[]`、机位图仍为 2 个 component。这是有效的失败证据，说明新搜索路径已运行但没有把
粗相似度分数冒充几何关系。84 张的 14 个 `grouping_accumulated_motion` split（累计位移
`0.020151–0.043758`）仍由未修改的 `0.02` 门禁产生，不能通过强制三张一组消除。Wenyuan 严格
几何继续保持 3/3 accepted、1 component，尺度为 `0.9999504/1.0020960/0.9997123`。

本轮因此只完成候选路径接线和 pair-level 验证，没有运行完整 84 acceptance，也没有把 84 标记为
已连通；下一步必须先找到能通过完整门禁的独立跨分量 pair，再进行一次完整 84 回归。

## 2026-10-22 完整 84 bridge 后回归

在 `17-24` pair 取得严格通过证据后，使用同一生产代码和真实 84 张 NEF 运行完整
`stack_acceptance_harness`。报告为
`/private/tmp/raw-editor-global-84-20261022/stack-report-7dbb6b21-e094-4017-91f5-c08ae9b1ce99.json`，
SHA-256 为 `e46d6514cf2857b9e38602bb2a9a4a1919a26249912a89f10e2230dbc969ae76`。

84/84 解码，28 个 station，无 isolated/undecodable，network=0；运行约 `2848.283s`，peak RSS
`12035244032` bytes，低于 `25769803776` bytes 门槛。global proposal 生成 `360` 条候选记录，
其中 `18→26` 通过完整 selector 并进入最终关系图；最终 `accepted=14`、机位图 `14 components`
（旧基线为 17 components），因此有实质改善但仍未连通。失败阶段为
`fitted=5`、`insufficient_bidirectional_matches=276`、`insufficient_measurable_patches=81`、
`insufficient_grid_coverage=1`、`polished_fit_failed=1`；尺度、空间支持和方向门禁保持原值。

由于 geometry disconnected，本次 `quality_gate.verdict=not_run`、`output.written=false`，不能把
84 张宣称为拼接完成。需要先校准 pair harness 的机位编号：它的 `17-24` 固定夹具实际是
`DSC_3757..3759` 与 `DSC_3763..3760`，在本次完整运行的动态 station 编号中对应 `18→26`；该边
已经在完整报告中以 `scale=0.9580825`、`support=0.28910`、`970` 个内点通过 selector。完整运行的
`17→24` 是另一组素材（`DSC_3719..3720` 与 `DSC_3697..3698`），其 `insufficient_measurable_patches`
不能与 pair 证据直接比较。下一步应以稳定的 source-member 标识重跑真实跨分量 pair，并核对
station 级 prior/crop/world 坐标传递和候选覆盖，而不是改 `[0.95,1.05]` 或其它验收阈值。

## 2026-10-22 Source_RAW 关系仅作机位搜索种子

为处理机位先验偏移，production prior-free orchestration 现在会把已验证的跨
`Source_RAW` 关系先按当前 `render_scale` 转换，再组合到两个 `Virtual_Tile` 的局部坐标，作为额外搜索种子；它不会直接写入
`tile_relations`。每个种子仍要经过 Virtual_Tile 金字塔双向测量、polish、多模型 selector 以及
现有 `[0.95,1.05]`、24 内点、3px、中间支持、方向、光度和凸性门禁，只有 `selection.passes` 才能
进入权威关系图或 prior repair。新增坐标组合单测和 `virtual_tile_pyramid_tests` 71/71 通过；
完整 84 张 acceptance 已在该代码上重跑，但当前仍不能把机位图标记为已连通。

## 2026-10-22 缩放修复后的 84 张回归

报告为
`/private/tmp/raw-editor-source-pair-84-20261022-scaled/reports/stack-report-3a596c42-9e2c-4563-8886-cee841764a79.json`，
SHA-256 为
`62174b29438b1560b9edd738013fbbeb57aefeb19b8cb837ffb82181173ba754`。84/84 解码且输入 SHA 前后一致，
28 个 station、network=0，peak RSS `13947895808` bytes，低于 `25769803776` bytes 门槛。候选 378 条，
其中 `source_pairwise_match=1`；该 source seed 经过完整 selector 后以 `10→17` 进入 accepted，尺度
`0.963318`、空间支持 `0.252248`、中位误差 `1.668px`。但 `10→17` 已存在于旧基线，未带来新的跨
component 连接；最终 accepted 仍为 14，机位图仍为 14 components，`quality_gate=not_run`，
`output.written=false`。日志中的 `6→13` source/global 候选仍因空间支持不足被拒。此次结果证明缩放
修复后的 source seed 已接入真实 84 流程并受原门禁约束，但还没有找到新的硬 bridge。

## 2026-10-22 多层一致性 coarse seed 回归

在上述 source seed 之后，又允许带有多层候选一致性证据的 coarse bridge 作为 Virtual_Tile
搜索 basin；普通 coarse/sequence bridge 仍被排除，最终关系仍必须通过同一 selector 和
`[0.95,1.05]` 门禁。完整 84 张报告为
`/private/tmp/raw-editor-source-pair-84-20261022-consensus-coarse/stack-report-c6dde528-b342-4f55-819f-022b31e4de46.json`，
SHA-256 为 `276786b77fe7ca45beab0a95e33e5ba09e4ae1a08adc3ce0532857a769150bc6`。

本轮 84/84 解码、28 个 station、378 条候选；`accepted=14`、`rejected=364`、
`unmeasured=360`，连通分量仍为 14，成员数为 `[2,3,3,6,7,7,6,9,4,13,3,4,7,10]`。
seed 计数为 `global_coverage_similarity=363`、`rendering_prior=13`、
`prior_free_features=1`、`source_pairwise_match=1`；唯一 source seed `10→17` 仍通过
selector，但它已经存在于旧基线，没有形成新的跨 component bridge。由于几何图仍断开，
`quality_gate=not_run`、`output.written=false`，不能把 84 张标记为完成。

## 验证命令

本次代码变更的定向验证：

```text
npm run image-stack-preview:check
npm run typecheck
cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check
CARGO_NET_OFFLINE=true cargo check --manifest-path src-tauri/Cargo.toml --lib --offline --locked
CARGO_NET_OFFLINE=true cargo test --manifest-path src-tauri/Cargo.toml --lib --offline --locked \\
  panorama_stitching::virtual_tile_pyramid_tests -- --nocapture
# source-pair coordinate tests: 2 passed
# virtual_tile_pyramid_tests: 71 passed, 0 failed
git diff --check
```

`npm run focus-stack-quality:check` 的 Tone_Harmonizer 环境读取源码契约已在 2026-10-22 recovery
中修复：test-only diagnostics 目录读取移到 `cfg(test)` support，production `tone.rs` 不再读取环境开关；
该合同与全部 Node quality/preview contracts 已通过。最新 84 张报告仍以 `quality_gate` 和单一连通
机位图为准，当前没有达到这两个条件。

84 张长时间验收需要用户本地 RAW，不作为常规 CI 测试；raw report/log 输出由 `.gitignore` 保持
在仓库快照之外，仅以上述摘要和 SHA-256 追踪。

## 2026-10-22 Wenyuan Tone_Harmonizer recovery

中断现场的 source snapshots、shell history、Stack Reports 与 final-world diagnostics 已恢复并逐项
对照。恢复前工作树确实是四轮空间残差 `converged-1`：tone boundary max/median/P95 从 baseline 的
`18.504616/0.535470/2.180269` 改善到 `1.416714/0.007195/0.062777`，453 个窗口无 violation；
checkerboard held-out 161613 samples 的 median/P95 也从 `0.796468/6.833978` 降到
`0.033508/0.201460`。

该方案仍不得保留：正式 output-to-owner-source `roi_delta_e00` 从 baseline 的 256/256 pass、
median/P95/max `0.935631/1.358032/1.568372` 退化为 24/256 fail、
`1.223310/3.109732/8.189209`。这证明 overlap holdout 只能约束 tile 间一致性，不能阻止两个 tile
共同偏离真实 Source_RAW。空间场 production path 因此显式禁用；没有改变 1.5、2.0 或任何其它门槛。

禁用后两次 fresh Wenyuan 均精确复现安全 baseline：3/3 relation、1 component、8792×8698，
10/10 RAW unchanged、network=0，三种输出 hash 为 `a4b50f05…b2deaf` / `beaa008b…3f879` /
`42146a3c…1dd0`；gradient 256/256、noise/effective/local-scale/ROI 均 pass，owner 仍 `0.745873`，
tone 仍 `degraded/18.504616`。Langyuan fresh preview 保持 `c25fb9d…99c6`，2 seam plans 与 48876
changed-owner pixels 不变。完整 variant、relation/window Lab/linear RGB、gain/bias、cycle/residual、
holdout、hash 与回归证据见
`.agents/tasks/wenyuan-tone-harmonizer-recovery-results-2026-10-22.md`。

## 2026-10-23 Wenyuan owner Source_RAW fidelity oracle

在 production 空间场继续保持关闭的前提下，新增了 test-only current-run 绝对色彩证据。256 个正式
512px ROI 仍由本轮 Quality Gate 根据最终 owner/coverage 生成；另选 17 个 8px spatially stratified
owner fidelity cells，所有 cell 均位于 shared-world checkerboard fit 之外。每个 ROI/cell 只对应一个
owner Source_RAW，并复用 production 的 Residual_Warp inverse、
`inverse(tile_to_world * member_to_anchor)` 和 bounded Catmull–Rom sampler，不读取旧报告中的 ROI
坐标或 verdict。

fresh Wenyuan oracle 中，90,476 个 overlap fit cells 与 161,601 个 checkerboard holdout samples 保持
严格分离。固定 shared-world 四轮场只有在 relaxation 至少 `0.994140625` 时才能把 453 个 boundary
window 的 max Delta-E 压到 1.5 以下；独立 owner fidelity 的 median/P95/max 不退化约束只允许
relaxation `0.0`。因此该候选族的离散 Pareto gap 为 `0.994140625`（步长 `1/4096`），不存在 joint
solution。full candidate 在正式 256 ROI 上为 9 fail，median/P95/max
`1.152005/1.639207/8.356272`；baseline 仍是 256/256 pass、
`0.935631/1.358032/1.568372`。oracle 选择 identity fallback，`TONE_SPATIAL_FIELD_ENABLED=false`
未改变，也没有进入 production variant 或 retention 流程。

本轮离线 harness 1/1 pass，3/3 accepted relation、1 component、10/10 RAW SHA-256 前后一致、
network=0；TIFF/app JPEG/preview hash 仍为 `a4b50f05…b2deaf` / `beaa008b…3f879` /
`42146a3c…1dd0`。原始 test-only oracle SHA-256 为
`96517f29c7fc8103546b392416dc506bc3cd64e817c0e442d662fa0af1084d1a`；完整实现、证据和验证见
`.agents/tasks/wenyuan-tone-owner-fidelity-pareto-2026-10-23.md`。

## 2026-10-24 final-world Source_RAW ownership Phase 1

新增 test-only final-world oracle，把当次 station candidate sharpness/data/coverage 经最终 world origin、
Residual_Warp 和 Source_RAW inverse 映射到 16px final cells；generic coherent regions 只来自 source full
coverage 与低频 content-connected components，使用 544px evidence core、799px phase protection、8 个
互不重叠区域、reduced branch-and-bound 和独立 relaxation。Synthetic 9/9 与 Node contract 通过；模块
保持 `cfg(test)`，不读取 Quality Gate ROI/旧报告/fixture identity，也不改变 production output。

fresh Langyuan 的 1,173 options / 374 origins 可形成 8 regions，reduced owner estimate 为
`0.9993064`，但 exact compositor/source replay 只有 `0.8809950`，ROI=10，effective
`0.9789146` fail，boundary 出现 3px fail。fresh Wenyuan 的 564 / 188 options 也可形成 8 regions，
reduced estimate `0.9913682`，exact replay 却只有 `0.8526492`（station 1 `0.6752864`）、ROI=8；
Wenyuan strict geometry 保持 3/3、boundary 0px，effective/gradient/Delta-E pass。两次运行均
network=0、10/10 RAW digest unchanged。

因此本阶段不能进入 production：candidate cardinality 足够，但 16px reduced cell 跨 station ownership
cell、局部 registration displacement 与 sampling phase 后会高估逐像素 owner evidence，并且 Langyuan
还暴露 boundary/effective 回退。完整 Pareto 数值、artifacts 和最小后续决策见
`.agents/tasks/final-world-source-raw-owner-phase1-results-2026-10-24.md`。

## 2026-10-24 final-world Source_RAW ownership exact X0/X1

Phase 1 的 16px 中心分数已在 test-only oracle 中替换为逐 final pixel 证据。每个像素复用 production
world/residual/source/station 映射和百分位 shortfall 量化；只有 candidate provenance、station cell、
`f32` score bits、quantized decision、Coverage 与 seam membership 全同的横向像素才组成 sparse run，
每像素 Source_RAW sample coordinate bits 仍独立保留。Synthetic 13/13 和 Node contract 通过。

X0 真实窗口中，Langyuan 10 个完整 799px 窗、Wenyuan 53 个完整 128px 诊断窗（含 station 1）
均为 `prediction_replay_mismatch_pixels=0`。X1 随后逐像素复核全部正式 799px options：Langyuan
1,173 项中 762 项 exact-valid，全局乐观 owner 上界 `0.9969776550`；Wenyuan 564 项中 372 项
exact-valid，但即使忽略 overlap、把 8 regions 外 shortfall 全设为 0，仍至少有 797,443 个
shortfall pixels，owner 上界只有 `0.9867200611 < 0.99`。因此没有 replay output、运行 post-QG 或
接入 production；现有 fallback、Wenyuan 3/3 geometry/Tone fallback 与 Langyuan seam 均未改变。
继续前需要产品方选择 owner threshold 或 QG-aligned 799px coherence 约束。完整证据见
`.agents/tasks/final-world-source-raw-owner-exact-x0-x1-results-2026-10-24.md`。

## 2026-10-25 用户批准 Option B：QG-aligned owner regions

Requirements 11.18、design 与 task 20.12 已记录批准调整：owner solver 可在当前运行固定
`512px ROI + 16px ring + 256px stride` 网格上生成 544px coherent regions，不再保护任意相位的
799px span；owner 0.99、正式 ROI/criterion、单一 Source_RAW provenance 和独立 post-validation 不变。
Node contract 禁止读取 QG 已选 ROI、旧报告/verdict、fixture identity/name 或硬编码坐标。

native-warp-aware exact replay 中，Langyuan 用 8 regions 达到 owner `0.9988658574`，8 个正式 ROI，
local/effective/gradient/Delta-E 通过且 noise not-applicable，但新 source boundary 的可测最大偏差为
`4px`，违反不回归门禁。把“544px outer perimeter 当前已属于候选 source”作为通用 seam veto 后，
1,361 options 只剩 2 个 exact-valid regions。Wenyuan 无 veto 时 data baseline `0.9920119163`，但
8-region exact optimum 只有 `0.9865474345`；veto 后 633 options 中 0 个 exact-valid。最新 Wenyuan
no-replay 同时只有 2/3 accepted relation、7 ROI，不能替代此前 3/3 safe baseline。

因此 Option B 已实现和验证但不接 production；正常 build 保持 deterministic current-owner fallback。
完整证据见 `.agents/tasks/final-world-source-raw-owner-option-b-results-2026-10-25.md`。
