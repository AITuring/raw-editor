# 任务2：单次运行、回归矩阵和内存门

- 起点：`v1 29371c0c`；隔离工作区`.worktrees/task-2`，分支`lane/task-2/run-matrix`。
- 基础二进制：`base-2026-11-12`，SHA `39bfe6f2aecc8a7ebc154efb94ba4b1f1a26257c825a676a0e54e8cde9cef03b`；原始提交`c0c1e5a9037fb4f8a8f8b983802848feb667b9ba`。
- 修改仅涉及工具、预期数据、测试和文档；不修改生产图像算法、质量阈值或既有图像基线。
- 本次同一开发者完成实现和L0集成检查，未启动并行agent。
- 状态：工具、快速层、中等层及负向验证均通过，L0集成验收完成。

## 实现与数据合同

1. `run.py`自动选择run/exclusive槽，只运行冻结副本；`execution.json`和`analysis.json`分开保存过程与判断。现有目录不可复用，不提供更新像素基线的参数。
2. 输入使用与Node runner一致的全局排序、大小写不敏感排除、子集截取及倒序。核对原文件前后摘要、完整参与/预期排除集合及Stack_Report中的源摘要。
3. RSS每0.5秒采样runner进程树；峰值取采样、原生getrusage和Stack_Report的最大值。超过24GiB或采样失败立即停止命令组；缺失原生usage/采样也判资源失败。20GiB是共享run槽的调度门，不与24GiB实际验收门混用。
4. 构建争用按整个运行区间判定；捕捉运行中才开始又结束的build。独占运行不把自身build占用记为争用。
5. 兰亭取景按每次运行隔离纸面审计、owner/coverage诊断与严格/回归保护；不更改审计抽样或保护阈值。
6. `regression-matrix.json`保存范围化预期。quick/medium/extended/full独立选择；known failure匹配只代表回归一致，不代表质量接受。来源、冻结副本和内存门不能通过known failure豁免。
7. 每次不符最多独占重跑一次，两轮均保留。仍不符返回1；恢复一致标`pass_after_exclusive_retry`。矩阵取消时停止当前任务，禁止继续或重跑。

## 工具测试

`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/stack-lab -v`：32通过、0失败，26.132秒。原任务1的17项也在本次完整重跑中通过。

新增覆盖：单次CLI结果、未知范围独占、输入排序及排除、摘要完整性、重复/辅助报告、内存等于/超过24GiB、usage缺失/无效、采样失败、超限停止、known failure不被标成质量接受、错误预期独占重跑与保留、独占后恢复、单次/矩阵取消及进程清理。

工具测试使用隔离临时Git仓库和模拟图像runner，不计入真实素材验收；nice、ps和信号行为在允许这些操作的环境测试。最终日志`output/acceptance/task-2/tool-tests-final.log`；早期通过日志另保留。

## 真实验证命令

```sh
python3 scripts/stack-lab/matrix.py --tier quick --binary base-2026-11-12 --lane task-2 --label r1
python3 scripts/stack-lab/matrix.py --tier medium --binary base-2026-11-12 --lane task-2 --label r1
python3 scripts/stack-lab/matrix.py --tier medium --binary base-2026-11-12 --lane task-2 --label negative-r1 --case nine-cold --matrix /Users/dp/Desktop/raw-editor/output/acceptance/task-2/negative-expectations.json
```

负向文件是正式矩阵的独立副本，仅将nine-cold的预期TIFF SHA改为64个0。正式预期和已验收基线不写回；负向矩阵预期返回1，并证明执行了第二次exclusive真实运行。

快速层7项命令全部通过：fmt/check、冻结库测试1093通过/0失败/44忽略，quality/datasets及runner/兰亭guard的Node检查。cargo check保留190项既有warning，不宣称Clippy或发布验收通过。

中等层5/5通过，均无需意外重跑：

| 范围 | QG | TIFF | 峰值GiB | contended |
| --- | --- | --- | --- | --- |
| nine-cold | pass / 8 ROI | 与冻结预期一致 | 8.896 | 是 |
| nine-hot | pass / 8 ROI | 与冻结预期一致 | 9.505 | 否 |
| langyuan-10 | pass / 8 ROI | 与冻结预期一致 | 9.557 | 否 |
| wenyuan-10 | pass / 8 ROI | 与冻结预期一致 | 10.371 | 否 |
| lanting-framed | pass / 8 ROI | 与冻结预期一致 | 9.200 | 否 |

9张冷/热TIFF=v28；阆苑10、文苑10、兰亭45 TIFF均与base-p1冻结值一致。兰亭纸面跨owner阶跃P95=4，最大7；strict和regression-only均为`quality_and_regression_pass`，没有放宽合同。7次实际运行（含两次负向运行）峰值最大10.371GiB，低于24GiB。

最后使用最终版本的分析与比较函数，对全部7份原始产物重新核对来源、二进制、runner/manifest、TIFF摘要、资源门和兰亭保护；5项正式预期一致，负向两轮仍仅有注入SHA差异。原始报告、分析和失败记录未覆盖；复核记录`output/acceptance/task-2/final-verification.json`。运行时保存的预期快照中，所有实际选择的case与最终矩阵完全一致。

机器摘要：[task-2-run-matrix.json](../benchmarks/task-2-run-matrix.json)。

负向验证已通过：命令按预期返回1，矩阵状态fail。两轮实际TIFF均为v28的`718a903d…f1bb47e3`，QG与资源门通过；每轮都只有被注入的`output_sha256`差异。第一轮为run/Rayon5，第二轮为exclusive/Rayon8并同时占住build，峰值分别9.926/9.137GiB。两轮原始分析与错误预期均保留，不将负向矩阵fail误记为素材质量失败。

## 证据与范围

- 本地产物分别位于`output/acceptance/task-2/matrix-r1-quick/`、`matrix-r1-medium/`、`matrix-negative-r1-medium/`。原始报告、日志、执行记录和预期快照保留，不覆盖首轮失败。
- 9张冷/热是独立进程与独立alignment cache；未清空OS页缓存。快速层与部分真实运行重叠，按实际标记contended，耗时不作性能结论。
- extended与full完成配置/选择验证；本轮任务2的真实完成标准是中等层和错误预期重跑，不宣称重新运行祁序/枫鹰扩展层或未验收全量。
- 任务3模板/复看队列和任务4的“编译+运行+排队”完整演练未在本任务勾选；本轮5线程的输出证据按实际结果记录，不扩大到跨素材、跨线程或负载确定性结论。
- 原有`.vscode/settings.json`保持；主工作区的两个未提交草稿先备份到`output/acceptance/task-2/pre-existing-drafts/`，以本任务验证后的实现替换。
