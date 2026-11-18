# 文苑空源排除 report contract 设计

状态：设计完成，尚未进入正式 runner、manifest 或历史 baseline。依据任务5冻结报告和任务14 shadow audit。

## 目标

`DSC_4885.NEF` 是选中输入中的真实文件，但当前为 0 bytes，native decode 失败。报告需要同时表达“输入被登记”和“该输入不参与活动合成”，避免把空源静默当作缺失，也避免仅修改 `expectedSkipped` 伪造来源通过。

## 候选 report schema

```json
{
  "input": {
    "selected_source_count": 126,
    "active_source_count": 125,
    "excluded_source_count": 1,
    "sources": [
      {
        "path": ".../DSC_4839.NEF",
        "decoded": true,
        "sha256_before": "<input sha>",
        "sha256_after": "<input sha>"
      }
    ],
    "excluded_sources": [
      {
        "path": ".../DSC_4885.NEF",
        "reason": "empty_source",
        "decoded": false,
        "sha256_before": "",
        "sha256_after": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
      }
    ]
  }
}
```

`sha256_after` 必须等于该路径当前文件的实际 SHA；空文件的固定值为 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。上面的 `sha256_before` 保留 native 当前语义（空字符串），不把它改写成伪造的输入摘要。

## 验证规则

1. execution 输入集合仍为 126 张，且每个输入路径和摘要保持不变。
2. `sources` 只包含活动输入；路径集合和前后摘要对 125 张活动输入逐项相等。
3. `excluded_sources` 与 `expectedSkipped` 一一对应；每条必须有明确 reason、`decoded=false`、空 `sha256_before` 和实际文件 `sha256_after`。
4. `selected_source_count = active_source_count + excluded_source_count`，且三项计数写入 Stack_Report。
5. 原生质量门、tone、density 和 closure 仍独立判定；排除合同通过不授予 D2/D3。
6. 采用该 schema 后必须重新生成完整 0.25 baseline，并保存旧 baseline，不覆盖历史报告。

当前冻结报告的 shadow 结果是：严格合同失败，候选合同通过。机器审计见 [wenyuan-exclusion-contract-audit.json](/Users/dp/Desktop/raw-editor/output/acceptance/task-14/wenyuan-exclusion-contract-audit.json)。

## 启用前置条件

- 获得 `DSC_4885.NEF` 的非空原始文件，或由 L0 明确批准把现有空文件作为正式预期排除输入；
- 评审并实现 report schema 与 runner 的活动/排除集合判定；
- 用同一冻结二进制和输入摘要重生成 baseline，随后再决定是否进入 scale=1。


## 实际报告投影验证

脚本 [wenyuan_excluded_source_candidate.py](../../scripts/stack-lab/wenyuan_excluded_source_candidate.py) 将任务5实际 Stack_Report 只读投影到候选 schema。126 个选中输入被拆为 125 个活动源和 1 个排除源；活动路径/摘要、排除记录和计数闭合均为 `true`。这证明 schema 能表达当前报告，但不改变正式 runner 的判定。

机器摘要：[wenyuan-excluded-source-candidate.json](/Users/dp/Desktop/raw-editor/docs/benchmarks/wenyuan-excluded-source-candidate.json)。

## 最小 report 实现（待 runner 启用）

分支 `lane/task-14/excluded-report-schema` 已在 Stack_Report schema 8 中加入 `input.excluded_sources`。解码失败后，native 复制对应 `SourceRecord`（保留 `decoded=false` 和 `sha256_before`），终止写报告时对活动源与排除源共同执行 `sha256_after` 重检；原有 `input.sources` 保持不变，旧消费者不会因字段迁移而丢失记录。正式 runner、manifest 和 baseline 尚未启用该字段。

定向 Rust 报告测试 20/20 通过，`cargo check --tests --offline` 通过。完整真实素材回归仍待 runner/schema 评审后进行。

## runner opt-in 判定实现

`run.py` 已支持 schema-8 的 `input.excluded_sources`：保留旧 `input.sources` 记录后，先过滤明确排除路径，再对活动源做原有前后摘要相等检查；排除记录必须与非 subset manifest 的 `expectedSkipped` 一一对应，且 `decoded=false`、`sha256_before` 为空、`sha256_after` 等于 execution 摘要。没有 `excluded_sources` 的旧报告继续走旧逻辑。

合成 schema-8 runner 回归测试 20/20 通过；真实 Wenyuan manifest 尚未增加 `expectedSkipped`，因此正式入口仍保持拒绝，等待评审和重基线。
