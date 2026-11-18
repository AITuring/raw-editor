# 任务9：来源锁定修图设计

- 更新：2026-11-18；状态：已完成（设计文档）；分支：`lane/task-9/source-lock-design`；起点：v1 `7647bbf3`。
- 范围：来源锁定、区域撤销、导出合同、Stack_Report 审计和人工区域与自动 Quality Gate 的边界。
- 实现状态：只写文档；没有新增前端、Rust、Stack_Report schema 或生产默认行为。

## 设计原则

来源锁定修图只改变“某个世界坐标区域允许使用哪些已验证来源”的候选约束。它不直接写入 RGB，也不替换 owner、Coverage 或几何证据。重新渲染仍由现有 compositor 选择像素；如果指定来源不满足 Coverage、footprint、alpha 或几何约束，区域进入 `stale`/`unresolved`，系统不能静默换成另一来源。

锁定操作必须指向本次运行已经登记的来源记录。来源记录由绝对路径、输入 SHA-256、source id、station/capture group 和 source-to-world 变换组成。用户界面展示的 owner 数字必须同时显示来源文件名，不能把 station owner 或局部 legend index 当作全局 RAW owner。

人工区域用于定位和复核，不参与自动 Quality Gate 的统计。它仍然必须保留有效的 Coverage、owner、输入和操作证据；人工标记不能把透明像素变成有效像素，也不能把 QG 的 `insufficient_evidence` 改成 pass。

## 用户流程

1. 在来源/owner 覆盖层中点击像素或拖出区域。面板显示世界坐标、当前 owner、来源路径和 SHA、Coverage/alpha 状态、station/capture group、局部几何误差和是否可锁定。
2. 用户选择“锁定此来源”或“允许来源集合”。单来源锁定适合笔画、印章等关键细节；集合约束适合纸面或低纹理区域。提交前显示将受影响的像素估计、当前 source/coverage digest 和失效条件。
3. 用户可添加“人工复核区域”标签。该标签只影响复核视图和 QG 统计排除清单，不创建来源约束。
4. 每次提交形成一个可撤销命令。撤销只回退最近一次已提交命令；重新执行新命令会清空 redo 分支。拖动中的临时预览不写入历史。
5. 导出前显示锁定区域、人工区域、未解析区域和 stale 区域的数量及面积。存在 stale/unresolved 锁时，导出必须明确选择“保留失败证据”或取消；不能自动降级为普通渲染。

## 可锁定条件

来源只有同时满足以下条件才可进入 `eligible_sources`：

- source manifest 中存在路径、输入 SHA-256、唯一 source id 和有效 `source_to_world`；
- 请求区域的每个采样 tile 都有 Coverage，且 owner/alpha 不是 `NO_OWNER`/透明；
- source footprint、prepared-image 边界和逆映射有效；不允许用图像矩形替代 footprint；
- 运行版本、输入 manifest、画布原点/尺寸、坐标系和变换指纹一致；
- 没有未解决的来源完整性或报告 digest 错误。

锁定区域使用世界坐标 polygon 加 tile margin，而不是保存屏幕像素。保存时同时记录 `coordinate_frame_id`、world bbox、polygon、requested source ids 和 selection policy。重新取景、输入 SHA、canvas origin、pose 或 source-to-world 变换改变时，相关锁全部标为 `stale`，不得继续影响渲染。

## 数据合同

锁定文档是独立 sidecar，不改变既有 Stack_Report 顶层 schema：

```json
{
  "schema": 1,
  "run_id": "native-fixture-…",
  "input_manifest_sha256": "…",
  "coordinate_frame": {"id": "world-v1", "canvas": [31090, 11008], "origin": [2607, 875], "units": "native_output_pixels"},
  "source_bindings": [{"source_id": 42, "path": "/…/DSC_3687.NEF", "sha256": "…", "station_id": 7, "capture_group_id": 3, "source_to_world_sha256": "…"}],
  "locks": [{"lock_id": "lock-0001", "status": "active", "region": {"polygon_world": [[120,80],[640,80],[640,500],[120,500]], "bbox": [120,80,520,420]}, "requested_sources": [42], "policy": "single_source", "eligible_tiles": 17, "selection_digest": "…", "created_event": 4, "stale_reason": null}],
  "manual_review_regions": [{"region_id": "review-0001", "polygon_world": [[…]], "reason": "paper_seam_review", "excluded_from_quality_gate": true, "created_event": 5}],
  "history": {"head": 5, "events": [{"event_id": 5, "kind": "add_manual_review_region", "inverse_kind": "remove_manual_review_region", "before_digest": "…", "after_digest": "…", "coordinate_frame_id": "world-v1"}]}
}
```

`selection_digest` 对排序后的 source id、region polygon、Coverage/owner tile digest、pose/source-to-world 指纹和 policy 做 SHA-256。它不包含 RGB，因此锁定 sidecar 可复核但不能伪造像素证明。

## 撤销与并发

事件日志采用单调 `event_id`，每个命令保存前后状态摘要和逆操作。撤销生成新事件，不删除历史；导出包保留完整事件链。一个编辑会话只能写入自己声明的 `base_run_id` 和 `coordinate_frame_id`；检测到另一个会话或新渲染版本改变 head 时，提交必须失败并要求重新载入。合并两个会话不做隐式 last-write-wins，需由用户选择区域或放弃一方。

锁定约束优先级为：`stale/unresolved` > `single_source` > `allowed_sources` > 普通自动 owner。高优先级状态只能阻止或缩小候选集合，不能绕过 Coverage/alpha/footprint 验证。区域外的自动 owner 选择完全不受锁定影响。

## 导出与 Stack_Report 绑定

导出目录必须同时包含原生 TIFF、`source-locks.json`、`source-locks.events.jsonl` 和 Stack_Report 的只读引用摘要：sidecar SHA、active/stale/manual 区域计数、锁定 tile 数、QG 排除 tile 数。

导出前后必须核对 input/reference/source manifest、owner、Coverage、canvas、alpha 和 bit depth。锁定 sidecar 改变而 TIFF 未重新生成时，导出标为 `stale_export`；TIFF 改变而 sidecar 未更新时，导入复核标为 `binding_mismatch`。两者都不能被显示为已确认成片。

自动 QG 报告同时给出 `all_tiles` 和 `auto_eligible_tiles` 两套计数，并记录人工区域排除的 tile id、原因和面积。人工区域不能删除失败 criteria；当排除导致样本不足时，verdict 必须是 `insufficient_evidence`。

## 验收清单

- 锁定只引用真实 source manifest，不能输入裸路径或手写 owner 数字；
- 缺 Coverage、footprint、alpha、digest 或变换时锁定失败或 stale；
- 撤销/重做保持事件顺序、状态摘要和坐标系；并发 head 冲突拒绝提交；
- 重新取景、重跑或输入改变后旧锁不会继续影响新像素；
- 人工区域从自动 QG 排除但保留 coverage/owner/source 证据；样本不足不通过；
- 导出 TIFF、sidecar、事件日志和 Stack_Report 引用互相绑定；任一 SHA 不一致都拒绝发布；
- 没有用户明确确认时，锁定操作不会被解释为 D4/D5。

后续实现应先在独立前端/诊断区域加入 sidecar 读写和可视化，再接入 Rust 只读约束；不能直接把本设计变成全局 owner 或质量门槛修改。
