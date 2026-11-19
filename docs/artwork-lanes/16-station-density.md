# 任务16：马远2 / 睡觉输出密度与闭环诊断

- 更新：2026-11-19；状态：**unit-contract 与 station/source probe 审计完成，单变量 factor A/B 尚未开始**。
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

## 下一步 bounded brief

若继续运行，控制组和实验组必须使用同一冻结 binary 与输入摘要；任何实验仍需经过原有 output-pixel closure 门。0.25 结果不能外推原生 D2/D3。
