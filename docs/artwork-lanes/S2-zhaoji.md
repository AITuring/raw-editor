# S2-zhaoji：赵佶任务17新素材

- 更新：2026-11-23；状态：**diagnostic_rejected**（scale=0.25，独占，preview-only）。本卡记录 D1 的原生失败链，不宣称 native/QG 通过。
- 输入目录：`/Users/dp/Downloads/书画拼图/赵佶`；扩展名：`.nef`；源图数：47；字节数：`2703135452`。
- 源集合 aggregate SHA-256：`d839174b01b7ce9ee5e04d9bc2070f3e83d4047c7ab8bf59e327bebaaa2079a8`；manifest：`tests/acceptance/focus-stack-datasets.json`；D1 预检 47/47 解码、SHA 未变化，source-contract 预检通过；native 结果行未生成，不能把中止前的 streamedSources=null 当作完整流式参与证明。
- 冻结二进制：`base-2026-11-12`；binary SHA-256：`39bfe6f2aecc8a7ebc154efb94ba4b1f1a26257c825a676a0e54e8cde9cef03b`；不改生产算法、阈值或 bounds。

## D1 运行记录

- 运行目录：`output/acceptance/task17/zhaoji-d1-r1/`；结构化摘要：`output/acceptance/task17/zhaoji-d1-r1/zhaoji-s0p25-task17-zhaoji-d1-r1.summary.json`；Stack_Report：`output/acceptance/task17/zhaoji-d1-r1/zhaoji-s0p25-task17-zhaoji-d1-r1-stack-reports/stack-report-a5b8f23a-46a1-4798-b21c-b68ef4fa659c.json`。
- 主路径：`layered_virtual_tile`；13 stations，1 个连通组件，isolated=0；topology ambiguous stations `[3,4,10,11]`，但未形成结构断连。
- closure：`converged`，参与约束 36，residual median/P95 `0.146791/1.705441px`，max direct-pair P95 `2.7683988px`；该项不是本次拒绝根因。
- output density：**rejected**；factor `1.29296875`，direct P95 `3.5794531px`，residual median `0.1897961px`，corner `1.5572392px`，`bypass_reason=output_pixel_closure_limit_exceeded`，worst station=12/source index=46。
- native harness：exit 101（`focus_stack_fixture.rs:632`）；没有输出图、artifact validation、tone、boundary 或 QG 结果，因此不能进入 D2/D3。摘要与审计 JSON：[`task-17-zhaoji-diagnostic-r1.json`](../benchmarks/task-17-zhaoji-diagnostic-r1.json)。
- memory gate：**PASS**；RSS 峰值 `12,451,872,768` bytes，阈值 `25,769,803,776` bytes，8 threads；运行约 679.3s。

## 诊断边界

本轮已确认输入预检、selected path、连通性、closure、density 和 memory gate；tone、boundary、owner、QG 因 density 闭合拒绝而未进入。下一步只在有明确的单一 source/geometry 假设时做 bounded、test-only 实验；不修改生产阈值、bounds、fallback、pose 或 manifest 排除规则，也不重复无假设的全量长跑。

## 单一 station Jacobian bounded A/B brief

由 D1 的最坏组 station 12 / `DSC_4299.NEF`、minimum scale `0.7739037` 定义唯一实验变量：对 station 12 的 `tile_to_world` 在 tile 中心保持位置，施加 isotropic factor `1.29296875`。控制组与实验组固定同一 47 张输入、同一 test-only binary `task-16-station15-17-jacobian-r1`（SHA-256 `c809bbfed9f89ee1109d0e48caacad5a81dd9f5aadd9b7f8500ed0c2ba0d67a8`）、同一 scale=0.25、8 线程和 24 GiB 内存门；不扫描其他 factor。实验合同见 [`task-17-zhaoji-station12-jacobian-brief-r1.json`](../benchmarks/task-17-zhaoji-station12-jacobian-brief-r1.json)。

## station 12 bounded A/B 结果

- 控制组与实验组均为 47/47 预检通过、13 stations、closure `converged`、8 线程、内存门通过；两组都在 density closure 前退出，tone、boundary、QG 未运行。
- 控制组 density factor `1.29296875`、direct P95 `3.5794531px`，最坏 station/source 为 `12/46`；实验组施加 marker `station=12 factor=1.29296875` 后 factor 降至 `1.244140625`、direct P95 `3.4442774px`，但仍超过 3px 门，最坏组转为 station `10` / source `32`。
- density probe 显示 station 12 的四个 source（43–46）均处于 `0.774–0.780` minimum scale，因此问题不是单一 `DSC_4299.NEF` 内容；但 station 12-only 修正仍不足以闭合全局 output lattice。完整 A/B：[`task-17-zhaoji-station12-jacobian-ab-r1.json`](../benchmarks/task-17-zhaoji-station12-jacobian-ab-r1.json)。
- 结论：停止 Jacobian factor 扫描，不推广 test-only 修正；只有提出新的多站点或 source/geometry 假设时才重开。
