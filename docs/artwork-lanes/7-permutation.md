# 任务7：阆苑置换检验离线实验

- 更新：2026-11-18；执行状态：已完成（test-only）；分支`lane/task-7/permutation`，起点v1 `711e2240`。
- 目标：借鉴ASHLAR，用确定不相邻机位对估计随机NCC高分位；只写离线证据，不改生产配准、Stack_Report、质量门槛或矩阵。
- 输入：阆苑84张冻结baseline的只读`source-alignment.json`（SHA `ab22db8a9ac12826a0d45d8854f8f998e94923f8814dda212f9805c97195d142`）及84张只读代理图（400×267）。原生0–2证据来自保存的三张station plane与`joint-production-input.json`。
- 实验脚本：[permutation.py](../../scripts/stack-lab/permutation.py)；报告：[机器摘要](../benchmarks/langyuan-task-7-permutation.json)。原始输出：`/Users/dp/Desktop/raw-editor/output/acceptance/task-7/langyuan-permutation-r4.json`。

## 方法

从保存的跨组已接受边中选76条（inliers≥15、capture-group距离≤1），使用每条真实边的direct homography固定采样几何；将目标来源替换为capture-group距离>1的确定伪随机来源，seed=`8762272649496`，重复1000次。真实与置换都用相同38×25内部网格、400×267灰度代理和交错强度/水平梯度NCC；不写入pose或报告。

另外回放保存的Langyuan 0→2三站点原生关系：64 inliers、拟合误差1.2600018px、hull support0.3613。由于保存矩阵来自nalgebra扁平化，离线读取时明确转置其布局；原生plane代理NCC为0.7462/66 samples。该值不与84张代理分布直接混作生产阈值。

## 结果

| 分布 | 数量 | P50 | P95 | P99 | 最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 真实跨组边 | 76 | 0.981396 | 0.993236 | 0.993542 | 0.993542 |
| 非相邻置换 | 1000 | 0.884972 | 0.965887 | 0.976218 | 0.987301 |

随机P99为`0.976218`，真实边中`39/76`高于它。分离不完整，不能用这个代理实验替换现有固定门槛，也不能据此静默放行或拒绝某条生产边。0–2的原生门禁证据与代理NCC不一致，说明下一步应做原生patch-level置换或任务8 provider 对照。

## 验证与范围

- `python3 -m unittest -v tests/test_permutation_lab.py`：3通过、0失败；覆盖NCC恒等/反相关、投影无效深度和分位数空输入。
- r3/r4同seed报告逐字节一致；84张代理源SHA和alignment SHA固定。
- 未修改Rust、生产匹配、Stack_Report、QualityGate、阈值、回归矩阵或任何素材基线。
- 任务7结论：**完成离线诊断，未证明可迁移的生产阈值收益**。下一步任务8：注册 provider 离线对照。
