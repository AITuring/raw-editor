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

## 下一步 bounded brief

若继续运行，只能固定一个 station/source density 假设，先在同一 scale=0.25 范围保留原 density 证据，比较基础 closure、输出格 closure、最坏探针、relation/support、来源合同和 memory gate。当前 probe 已足以锁定最坏组；下一步应针对 station 17 的单一 source/Jacobian 假设写 brief，再决定是否值得 A/B。控制组与实验组必须使用同一冻结 binary 和输入摘要；任何实验仍需经过原有 output-pixel closure 门。0.25 结果不能外推原生 D2/D3。
