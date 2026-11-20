# 任务16：马远2 / 睡觉输出密度与闭环诊断

- 更新：2026-11-20；状态：**scale=1、boundary、owner fallback 与 residual density 诊断已完成；native acceptance 仍拒绝，生产路径不变**。
- 范围：马远2 47 张、睡觉 45 张，沿用 `base-2026-11-12`、scale=0.25 的任务5诊断；不改 pose、density 公式、3px 门限、输入 manifest 或历史基线。
- 依赖：任务5.1、5.2；与任务13/14/15冲突。任务15未触发，因为两套运行均正常结束且峰值约12–13GiB，低于24GiB内存门。

## 固定失败链

- 马远2：基础 closure direct-pair P95 `2.6046616px`，density factor `1.1767578125` 后为 `3.0650559px`，超过现有 `3px` 门；最小 native scale `0.8504955`，最坏 station/source 为 `17/33`、station point `[2049,33]`。
- 睡觉：基础 closure direct-pair P95 `2.5386065px`，density factor `1.37890625` 后为 `3.5005004px`，超过现有 `3px` 门；最小 native scale `0.7253261`，最坏 station/source 为 `14/21`、station point `[129,1345]`。
- 两套 residual median 与 corner correction 乘算后仍低于各自门限，当前决定性阻断是 direct-pair P95；不能把基础域 closure pass 当作输出格 closure pass。

## 只读 unit-contract 审计

脚本 [task16_density_contract_audit.py](../../scripts/stack-lab/task16_density_contract_audit.py) 从冻结诊断和源码常量重算基础 closure、output-lattice closure 及允许 factor 上限。结果见 [task-16-density-contract-audit.json](../benchmarks/task-16-density-contract-audit.json)：

- 马远2 在 3px direct-pair 门下的允许 factor 上限为 `1.1517811`，实际 `1.1767578`；
- 睡觉允许 factor 上限为 `1.1817507`，实际 `1.37890625`。

这只是公式审计，不是成片或原生验收；不能通过截断 factor、跳过 density、放宽 3px 或绕过 closure 来制造通过。

## 马远2 station/source probe 导出（r3）

在同一 scale=0.25、`task-16-density-diagnostics-r3` test binary 下，导出了 18 个 station、47 个 station/source 组、124,543 个 native density probes，并同时保存 station→world pose。运行仍因原有 output-pixel closure 拒绝，QG 未运行；内存门通过，峰值 RSS `11,102,814,208B`。

probe 复算与报告一致：最坏组为 station `17` / source `33`，point `[2049,33]`，minimum world-per-source scale `0.8504955088`，p01 `0.8551132`，median `0.8892349`。下一个组 station `17` / source `34` 的 minimum 为 `0.8523744`。这证明当前 factor 来自真实 station→world 与 source→station Jacobian 采样，不能用记录格式或闭环单位错误解释。

机器摘要：[task-16-density-probe-audit.json](../benchmarks/task-16-density-probe-audit.json)；原始 test-only probe 目录位于 `/Users/dp/Desktop/raw-editor/output/acceptance/task-16/mayuan-2-mayuan-2-density-diagnostics-r3/diagnostics/density`。

## station 17 单一 Jacobian bounded brief

source33 与 source34 在 station17 同时出现约 `0.85–0.89` 的 world-per-source scale；source33 的 `source_per_station` 在最坏点为单位矩阵，source34 也接近单位矩阵。因此单一假设定义为：station17 的 `tile_to_world` isotropic Jacobian 在 station 中心附近被低估，实验若进行只能 test-only 修正 station17 pose/Jacobian，不改 source sampling、density factor 或 closure 门。

可行性先验已经足够决定是否运行：若 station17 最低 scale 被完整抬到 1，下一最坏 station16/source46 的 minimum scale 仍为 `0.8600202`，量化后的 density factor 为 `1.1630859`，基础 direct P95 `2.6046616px` 会预测为 `3.0294453px`，仍超过 3px。也就是说 station17 单变量即使理想修复，也不能让 output-pixel closure 通过；继续做全量控制/实验只会重复失败链。

机器 brief：[task-16-station17-brief.json](../benchmarks/task-16-station17-brief.json)。决定：**不启动 station17-only 的控制组/实验组**；若要继续，必须另写包含 station16/15 的多站点假设，不能把它混进当前单变量 brief。

## station 15–17 shared Jacobian bounded brief

probe 的低尺度组连续落在 station15/16/17：minimum 分别为 `0.8829276`、`0.8600202`、`0.8504955`。因此下一实验定义为一个共享变量：对这三个 station 的 `tile_to_world` 在各自中心保持位置、统一乘 isotropic factor `1.1767578125`；不改变 source sampling、relation 阈值、density 公式或 3px 门。环境变量为 `RAW_EDITOR_TEST_STATION_JACOBIAN_SCALE=15,16,17:1.1767578125`。

若该三站簇是共同的 Jacobian 低估，下一最坏组预计转到 station11/source32，量化 factor `1.1103516`，direct P95 预计 `2.8920901px`，理论上留出 `0.10791px` 的 3px 余量。因此这次**值得**做同 binary、同输入的 control/experiment；最终仍必须通过 closure、density、owner、source contract 和完整 QG，不能由理论预测授予通过。

机器 brief：[task-16-station15-17-brief.json](../benchmarks/task-16-station15-17-brief.json)。

## station 15–17 shared Jacobian A/B 结果

使用同一 `task-16-station15-17-jacobian-r1` binary、同一 47 张输入和同一 exclusive 合同完成 control/experiment。实验 marker 确认 station15/16/17 均施加 factor `1.1767578125`。

- control：density factor `1.17578125`，direct P95 `3.0407528px`，output-density rejected，QG 未运行。
- experiment：density factor `1.1103515625`，最坏转为 station11/source32，direct P95 `2.8920901px`；output-density applied、QG pass、TIFF/来源合同/内存门通过。
- 但 experiment 的 native acceptance 仍因 `local_scale_median` 拒绝（该 criterion 在 QG 中仍是 diagnostic/not_applicable），因此没有达到 D2/D3。实验峰值 RSS `12,517,195,776B`，实验输出 SHA `6dd5b883…`。

机器摘要：[task-16-station15-17-jacobian-ab.json](../benchmarks/task-16-station15-17-jacobian-ab.json)。结论：shared Jacobian 修正只证明可以越过 output-density/QG 表面门，不能推广到生产；不扩大 station 集合、不继续扫 factor，回到 local-scale native 根因诊断。

## minimal density-feasible factor brief

上一次 shared factor `1.1767578125` 使 density/QG 通过，但 native `local_scale_median` 被过度校正后拒绝。基于同 binary control 的基础 direct P95 `2.5861552px`，3px 门允许 density factor 上限 `1.1600232`；station17 minimum scale `0.8508270` 只需共享 factor `1.015625`，量化后的 predicted density factor 为 `1.1582031`，predicted direct P95 `2.9952931px`。

这不是盲扫：只验证一个由现有 closure 上限推导出的最小可行 factor，目标是减少 local-scale 过校正，同时保留 output-pixel closure 门。brief：[task-16-station15-17-minimal-factor-brief.json](../benchmarks/task-16-station15-17-minimal-factor-brief.json)。

后续两档已完成并收口：`1.015625` 实测 density factor `1.1591797`、direct P95 `3.0192708px`，仍未过 closure；`1.0234375` 实测 factor `1.1503906`、direct P95 `2.9963783px`，但 QG 仍有 boundary 1 处失败且 native `local_scale_median` 仍拒绝。证据见 [task-16-station15-17-factor-followup-ab.json](../benchmarks/task-16-station15-17-factor-followup-ab.json)。因此 shared Jacobian correction 不能同时满足 density、boundary 和 native local-scale；停止 factor 扫描，不推广、不放宽门限，回到 local-scale native 根因 brief。

## local-scale native contract 与 scale=1 brief

代码合同明确：`run_gate.rs` 在 `acceptance_render_scale < 1.0` 时把 `local_scale_median` 标记为 diagnostic，`criterion_verdict` 必然返回 `not_applicable`；native acceptance 对该 criterion 不接受 `not_applicable`。因此 task16 的 scale=0.25 QG pass + native reject 是预期合同边界，不能继续在 0.25 扫 Jacobian factor。

若要验证 local-scale，下一步只能做 scale=1 native feasibility control，固定同一 47 张输入、同一 test binary、exclusive 槽和24GiB门；在 control 的 local-scale、density、closure、source contract、memory 全部有结果前，不启动任何 Jacobian experiment。brief：[task-16-native-scale1-brief.json](../benchmarks/task-16-native-scale1-brief.json)。

## 下一步 bounded brief

若继续运行，控制组和实验组必须使用同一冻结 binary 与输入摘要；任何实验仍需经过原有 output-pixel closure 门。0.25 结果不能外推原生 D2/D3。

## scale=1 native control 结果

使用同一 `task-16-station15-17-jacobian-r1` binary、同一 47 张输入、scale=1、exclusive 槽和 24GiB 内存门完成全量 control。运行约 `10014.76s`，峰值 RSS `10,928,963,584B`，内存门通过；输入、binary、source digest、RGBA16/sRGB/alpha TIFF 和 output contract 均通过，closure `converged`。

- native local-scale 已经可测且通过：`local_scale_median=pass`、`local_scale_pixel_ratio=pass`，因此 0.25 的 `not_applicable` 合同解释得到原生验证。
- native acceptance 仍拒绝：`boundary_stroke_alignment=fail`（8 个 ROI 中 1 个测量为 `3.0px`）、`owner_sharpness_coverage=insufficient_evidence`（163 个 `owner_reverse_lookup_unresolved`）、`effective_pixel_count=insufficient_evidence`，`mtf50_normalized=not_applicable`。
- output density 没有形成可审计的原生 closure：`status=bypassed`、`factor=1.0`、`source_count=42`、`sample_count=0`，原因是 `nonidentity_world_residual`；因此不能把这次当成 density 通过。
- report 保留 `493` 条 degradation：`station_relation_scale_out_of_range=35`、`intra_station_rejected_from_group=79`、`fusion_single_frame_degraded=33`，并有 5 个 topology ambiguous station。该结果不支持直接运行 station15–17 Jacobian experiment，也不推广到生产。

机器摘要：[task-16-native-scale1-control.json](../benchmarks/task-16-native-scale1-control.json)；完整 analysis、Stack_Report 和 TIFF 位于 run directory。bounded 假设与实验合同见 [task-16-native-root-cause-brief.json](../benchmarks/task-16-native-root-cause-brief.json)。当前只运行一次严格 boundary repair A/B 并导出 unresolved 坐标；density bypass 的 residual-aware diagnostic brief 见 [task-16-residual-aware-density-brief.json](../benchmarks/task-16-residual-aware-density-brief.json)。density 仍单独保留，除非出现独立、可证伪的假设，否则不再扫描 Jacobian factor。

## strict boundary repair A/B 与 unresolved 坐标结果

使用新冻结的 test-only binary、同一 47 张输入和 scale=1 完成一次 A/B。`RAW_EDITOR_FOCUS_BOUNDARY_REPAIR=1` 将唯一可测 seam 的误差从 `3.000px` 严格降到 `0.000px`，日志记录了同一 owner pair 和重采样像素；但完整 native gate 仍为 `insufficient_evidence/rejected`，boundary criterion 变成 `not_applicable`，owner unresolved 仍为 `163`，density 仍因 `nonidentity_world_residual` bypass。因此不重复 boundary repair，也不启动 Jacobian factor 实验。

test-only `RAW_EDITOR_FOCUS_OWNER_UNRESOLVED_DIR` 导出的 163 点全部是 `raw_owner_unresolved`，集中在 station13（124 点，local y=`5252.81–5471.70`）和 station14（39 点，local y=`5234.35–5448.85`）的下部区域；其余 transform/out-of-bounds/station/evidence 计数均为零。这个空间聚类支持一次可证伪的 5×5 owner-stencil A/B，但不能预先假定它会通过 owner gate。

机器摘要：[task-16-native-boundary-repair-owner-export-ab.json](../benchmarks/task-16-native-boundary-repair-owner-export-ab.json)；原始坐标文件在该 run directory 的 `owner-unresolved/`。5×5 stencil A/B 已完成并将 unresolved 降为零；保留生产 3×3，不再叠加变量或重复 owner fallback，转回 boundary 可测性与 residual-aware density 合同。

## 5×5 owner-stencil A/B 结果

使用 `RAW_EDITOR_FOCUS_OWNER_STENCIL_RADIUS=2` 完成同一 47 源、scale=1、exclusive/24GiB 运行，boundary repair 关闭。163 个 unresolved owner 坐标全部归零，`owner_sharpness_coverage` 从 `insufficient_evidence` 变为 `pass`；默认生产半径仍是 1（3×3，最多 9 个候选）。

但 boundary seam 仍在 `(699,3621.5)` 为 `3.0px`，`boundary_stroke_alignment=fail`；`effective_pixel_count` 仍 `insufficient_evidence`，`mtf50_normalized` 仍 `not_applicable`，output density 仍 `bypassed/nonidentity_world_residual`。完整 native acceptance 仍拒绝，所以不把 5×5 fallback 推广，也不把它与 boundary repair 叠加再跑。

机器摘要：[task-16-native-owner-stencil5-ab.json](../benchmarks/task-16-native-owner-stencil5-ab.json)。结论：owner fallback 假设已验证为局部有效但不足以完成任务16；保留生产 3×3 合同，后续只处理 boundary 合同和 residual-aware density 诊断。

## residual-aware density 诊断结果

test-only 组成链已完成一次全量 scale=1 诊断：`station pose → ResidualWarp::forward_world_for_station → source_per_station.inverse()`。共 1,788,082 个可测样本、17 个 residual regions；结果只写 JSON，`plan_output_density` 仍 `bypassed`、factor 仍 `1.0`，没有修改 TIFF。

- composed scale minimum `0.0172509`、p01 `0.8607330`、median `1.0560082`；最坏为 station8/source18、station point `[8097,5409]`。
- 原始 closure 字段保留为 direct P95 `2.709115px`、residual median `0.318255px`、corner `122.816248px`；diagnostic 不把这些字段冒充 output-pixel closure，也没有输出可应用 factor。
- 这证明 residual bypass 是安全合同：实际残差场会产生强烈局部非恒等 Jacobian，不能把旧的 pose/source-only density factor 直接复用。完整 native acceptance 仍受 boundary/effective-pixel 等原有 gate 拒绝。

机器摘要：[task-16-residual-aware-density-diagnostic-result.json](../benchmarks/task-16-residual-aware-density-diagnostic-result.json)；原始 summary 和 18 个 station JSON 在该 run directory 的 `diagnostics/residual-density/`。任务16保持未验收，不进入生产 adapter。

## 任务16最终 bounded 决策

owner radius2 只在 test-only 路径把 unresolved `163→0`；生产仍保持 3×3。boundary repair 只把一个具体 seam 的局部误差 `3.000→0.000px`，但可测候选仍不足以让 native criterion 通过；residual-aware density 诊断证明 nonidentity residual 的 composed scale 最低 `0.0172509`，不能复用旧 factor。任务16仍不勾选、不推广生产、不继续 factor 扫描。

最终机器决策：[task-16-final-decision.json](../benchmarks/task-16-final-decision.json)。若继续，只能另写 boundary candidate-measurability 或产品合同；没有新的可证伪假设，不再启动全量 A/B。
