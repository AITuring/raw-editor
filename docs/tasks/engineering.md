# 工程、代码结构与发布任务

> [返回总入口](../project-progress.md) · 书画堆栈开发见 [tasks.md](../../.kiro/specs/layered-camera-group-focus-stitching/tasks.md)

本文件跟踪工程类任务：集成与发布（ENG）以及堆栈代码结构（原N0–N2）。2026-11-12从`docs/project-progress.md` §3.2–3.4和§6原样迁出（链接已按新位置改写）；以后在本文件勾选和追加。

<a id="engineering"></a>

## 工程、集成与发布（ENG）

- [x] **ENG.1** 2026-11-03已完成当时工作区Web类型/lint/build、翻译、macOS app及单张NEF导入→曝光→JPEG导出冒烟；实际范围见[固定基线与集成记录](../focus-stack-accepted-baseline.md)。
- [x] **ENG.2** v35冻结候选fmt/check、3项Node契约、6项覆盖/参考定向测试、release库测试1035 passed /41 ignored /0 failed；两套10张和同9张冷/热实跑已归档，失败如实保留。
- [x] **ENG.7** codex分支评估与移植（2026-11-12，用户要求先合入可接受的工作）：`codex/layered-focus-modularize-2026-10-26`（33个提交，基于2026-10-12的`7f7cf327`）中，15个模块拆分提交已被v1在2026-10-29的拆分取代（v1有523个`.rs`文件，该分支222个），文档提交已过时，5个提交的内容v1已有。移植3项v1缺少的修复：QG边界梯度模板要求整个5点模板同一来源且有覆盖（`1cc5e5a1`）；Virtual_Tile先验修复按每个成员自己的世界中心检查范围，整簇预检通过才提交，不再部分移动（`7e3a0488`修复部分及夹具`c2cab5e7`）；规范方向的边缘强度比不再按导入/分组顺序取倒数，非对称门槛与导入顺序无关（`fc22136f`）。5个局部接缝遍历提速提交（`27c1fd60`、`255f422f`、`ffc1fee5`、`bc5de888`、`b28d628a`）未移植：它们改写的是10-25版接缝循环，v1此后加入局部关系重采样带、128px有据重定位和8px清理，需要重新实现，归入PF.9。验收：库测试1093/0/44（新增7项）；9张冷/热TIFF=v28；阆苑/文苑10张TIFF与I1相同；兰亭45 TIFF与I1相同、QG pass/8ROI，严格保护通过；祁序/枫鹰0.1结果与I1g相同（QG fail，交L4/L5）。这些修复只在非对称门槛遇到反向导入、修复簇有越界成员、边界采样模板跨owner/覆盖边缘时改变决定，本轮回归集未触发。[机器摘要](../benchmarks/base-p1-codex-port.json)。
- [ ] **ENG.3** 完整桌面交互、不同相机素材、取消/失败恢复，以及Windows/Linux验收；2026-11-20当前 `typecheck`、`build`、`lint` 均通过，CI 有 Windows/macOS 构建矩阵，但仍缺当前桌面交互、取消/失败恢复和 Windows/Linux 实机验收。见 [ENG.3 audit](../benchmarks/eng3-desktop-acceptance-audit.json)。
- [ ] **ENG.4** 全特性严格Clippy、发布CI、签名/公证/分发；2026-11-20 strict clippy 仍失败（当前 all-targets 快照 505 条 diagnostics：lib 273、lib test 232，exit 101；Source_RAW 与三个未接入默认生产路径的契约模块已移入 `cfg(test)`，剩余共享/生产模块债务仍未清零），release workflow 有构建/发布矩阵但未发现 Apple/Windows 签名、公证或 Linux 分发验收。见 [ENG.4 audit](../benchmarks/eng4-release-audit.json) 与 [distribution audit](../benchmarks/eng4-distribution-acceptance-audit.json)。
- [x] **ENG.5** 当前新增WIP集成/提交范围已重新核对；从上次远程基线 `f582825f` 到当前 HEAD 的6个提交、20个变更文件均属于 N0.3/N1.11/N1.12/ENG.3/ENG.4 审计与 test-only 边界收口，工作树干净且 `origin/v1` 已同步。ENG.4 签名/公证/分发门仍单独保持 open。见 [ENG.5 scope audit](../benchmarks/eng5-scope-audit.json)。
- [x] **ENG.6** 两条代码线集成（2026-11-11新增，2026-11-12完成，S0）：WIP快照`ae5f5268`+`d981e20c`与`feat/stack-baselines`（`a9b17583`）在`integrate/i1`合并（`219910a5`），按用户选项A保留两条线行为，已快进合入v1并推送。集成中修复：QG重放逐帧曝光平面与预滤波（`c0f641bd`）；密度预滤波仅用于焦点扫描，站点tile保留金字塔，恢复阆苑0–2关系（`040c79be`）；多次拍摄回退原因及枫鹰panic（`33a8c646`）；导入前拍摄参数扫描，参数不一致（多文件夹、间隔>30分钟、机型/快门/光圈/ISO差>1%或不可读）用逐帧曝光平面，一致则统一调整，避免兰亭纸面偏暖（残差15/7/6→6/1/1，`83ccc1e3`）；QG边界描线只处理相邻区域对，修复枫鹰数小时卡死（`c8fd7e6a`）；runner接受流式合成的精确ownership行（`b68c9e22`）。验收：库测试1086/0/44；9张冷/热TIFF=v28；阆苑/文苑10张TIFF=v47；兰亭45 QG pass/8ROI；祁序/枫鹰0.1首次测得QG fail，用户选择严格合入，结果被对话框拒绝，交L4/L5。[I1机器摘要](../benchmarks/i1-integration-baseline.json)。

图片管理、一般局部编辑、批处理、参考图色彩迁移等功能入口见[产品README](../../README.md)。现有文档未给出这些模块完整端到端验收矩阵，统一纳入ENG.3，不凭产品功能列表标成“验收完成”。

## 堆栈代码结构（原N0–N2）

<a id="n0"></a>

### 3.2 N0：基线和工作区保护

- [x] **N0.1** 建立输入manifest、参考图digest、工作区快照和keep/adapt/test-only/exclude分流；v28已验收源码固定到独立本地分支，未覆盖当前WIP。
- [x] **N0.2** 恢复并保护兰亭45张legacy成片行为，保留不一致的裁切/质量诊断；2026-11-03完成编译、契约、库测试、Web检查、macOS构建和单图桌面冒烟的范围化集成。
- [x] **N0.3** 清理生产源码中的raw/preview/命令报告引用及实验artifact；2026-11-20完成三项 RAW display 开关、五项 focus/station/boundary 诊断、两项 ownership/gain 日志诊断的 test-only 边界收口；238个变量、366处读取经 parent/local `cfg(test)` 解析和逐项 review matrix 归属，41/41 shared/production 名称均明确为产品输出、资源预算、诊断证据或 legacy comparison control。legacy comparison 保持在默认 layered production path 之外，preview/Stack_Report/资源预算保留为产品契约。见 [N0.3 audit](../benchmarks/n0-3-production-artifact-audit.json)、[RAW env inventory](../benchmarks/n0-3-raw-env-inventory.json) 和 [review matrix](../benchmarks/n0-3-shared-review-matrix.json)。

<a id="n1"></a>

### 3.3 N1：生产代码职责拆分

- [x] **N1-1 / N1-2a / N1-2b** 共享类型、坐标、geometry/focus纯helper迁移，facade兼容重导出与回归完成。
- [x] **N1-3 / N1-4 / N1-5 / N1-6** report schema/recorder、pair候选、feature preparation/matching合同和匹配执行体拆分完成。
- [x] **N1-7 / N1-8 / N1-9 / N1-10** station solver/topology/closure、grouping、stitching、mosaic、processing和其他长模块迁移完成；模块地图已记录。
- [x] **N1.11** 处理仍超1000行的单函数/组件及建立import方向检查；14个超1000行源文件、5个超长声明和19个 child→facade 引用均完成决策矩阵：超长入口保留为有意职责边界，9个 production/shared import 判定为兼容 helper/facade 类型依赖，source-layout 已刷新且 stale=0。见[源码布局](../source-layout.md)、[N1.11 layout audit](../benchmarks/n1-11-layout-audit.json)、[import-direction audit](../benchmarks/n1-11-import-direction-audit.json)和[decision matrix](../benchmarks/n1-11-layout-decision-matrix.json)。
- [x] **N1.12** 后续每次职责迁移继续记录定向测试、完整回归和前后输出等价性；已建立机器可检 migration ledger，N1-1..N1-10 共10/10条记录具备三类历史证据。ledger 明确 `machine_replay=0` 且 `current_wip_applies=false`，当前或后续WIP不得套用历史等价性。见 [N1.12 audit](../benchmarks/n1-12-migration-evidence-audit.json) 和 [migration ledger](../benchmarks/n1-12-migration-ledger.json)。

<a id="n2"></a>

### 3.4 N2：测试支持与研究隔离

- [x] **N2.1** fixture/test_support、alignment、Virtual_Tile、acceptance、interpolation及properties按领域拆分；保留测试名、过滤路径、生成器和失败种子，历史迁移检查通过。
- [x] **N2.2** structural/owner/final-world oracle的依赖方向彻底收口；2026-11-20质量合同测试通过，静态依赖方向清单17/17通过，声明、导入及 facade 调用均保持在`cfg(test)`边界内；未将任何 oracle 提升为 release gate。见 [N2.2 audit](../benchmarks/n2-2-oracle-dependency-boundary-audit.json) 和 [dependency inventory](../benchmarks/n2-2-oracle-dependency-inventory.json)。
- [x] **N2.3** 将production invariant与research oracle契约分离；`stack-pair-evidence:check` 仍为 `--ignored` research harness，未进入 release/CI workflow；`focus-stack-quality:check` 为 production-only，research oracle 断言通过 `focus-stack-research:check` 显式 opt-in。2026-11-20两条合同命令均通过。见 [N2.3 release-gate audit](../benchmarks/n2-3-release-gate-audit.json)。
