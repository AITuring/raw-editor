# 任务16：马远2 / 睡觉输出密度与闭环诊断

- 更新：2026-11-19；状态：**unit-contract 审计完成，单变量重跑尚未开始**。
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

## 下一步 bounded brief

若继续运行，只能固定一个 station/source density 假设，先在同一 scale=0.25 范围保留原 density 证据，比较基础 closure、输出格 closure、最坏探针、relation/support、来源合同和 memory gate。控制组与实验组必须使用同一冻结 binary 和输入摘要；任何实验仍需经过原有 output-pixel closure 门。0.25 结果不能外推原生 D2/D3。
