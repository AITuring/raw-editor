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

