# 已验收堆栈基线与工作区集成验证

> [返回项目进度与任务总览](project-progress.md#acceptance) · 专项文档。本文保存冻结基线身份、保护方法和范围化集成证据；下文各轮状态为对应日期快照，项目最新结论统一见总览。

本次收口范围为用户确认的“原性能目标和既有质量门禁”，使用相同 9 张连续 NEF、scale=1。
它不代表全部 45 张或其他数据集已完成成片验收，也不要求把内容不适用指标改判为通过。

后续状态（2026-11-03 v31）：本文件的 v28 Git 基线与集成记录保持为历史快照。当前工作区已增加
默认预算下的无损后备复用，499 个 Rust 文件及新二进制另行冻结；同组 9 张冷/热再次通过，
输出 SHA 不变。当前验证见[性能优化文档第 32 节](focus-stack-performance-optimization.md)
和 [v31 验收摘要](benchmarks/sleeping-9-v31-bounded-backing-acceptance.json)。不能再把当前源码称为
逐文件等同于 v28；下文“本轮”均指 v28 固定和当时的集成验证。既有基线分支、当前分支及暂存区未改写。

后续状态（2026-11-04 v32）：当前工作区已修复 source seed 的 station-local/global 坐标混用，
并保留候选尝试历史，500 个 Rust 文件及二进制另行冻结。同组 9 张默认冷/热回归通过，输出 SHA
及原有来源/质量证据不变；阆苑/文苑 10 张仍因尺度门禁断连，不能据此升级跨素材验收。
详见[性能优化文档第 33 节](focus-stack-performance-optimization.md)和
[v32 复验摘要](benchmarks/cross-station-v32-local-source-seed-acceptance.json)。v28/v31 冻结基线保留。

## 基线身份与保护方式

- Git 父版本：`a0dc5f9c`（`v1`）。独立基线分支使用 `codex/accepted-stack-v28-baseline`，
  通过临时 Git index 创建；不切换当前分支，不改变当前 index，也不收走现有未提交修改。
  固定提交为 `504d904ff77781ac13e57f52c13fef8024b6c961`，包含 112 个变更路径；未推送远端。
  如需独立复核，可在新目录执行 `git worktree add /private/tmp/raw-editor-v28-review codex/accepted-stack-v28-baseline`，
  不应对当前工作区执行 reset/checkout 覆盖。本节后续集成结果是该固定基线之后的工作区记录。
- [源码清单](benchmarks/sleeping-9-v28-source-baseline.json)记录 497 个 Rust 源码文件的 SHA-256。
  本轮逐个复核，全部与 v28 冻结构建一致。
- 二进制 SHA-256：`e50f6a4c9f77b49ed20e6a20a36976a2fa4b3e5259d0598cd55aa729ab29a72c`。
  已从临时目录复制到本地持久归档 `output/integration/2026-11-03-accepted-baseline/accepted-release-binary` 并复核哈希。
- 输出 TIFF SHA-256：`718a903dcd0bf102fd7ea7f2be652168ad972f18bc540f5854225ee2f1bb47e3`。
- [验收摘要](benchmarks/sleeping-9-v28-final-acceptance.json)与冻结报告哈希已重新核对。
  详细产物仍在本地 `output/acceptance/sleeping-9-native-v28-final-acceptance/`。
- 本轮开始时的 status、tracked/index patch、忽略规则、237 个展开文件的清单、哈希和原内容压缩包，
  保存于 `output/integration/2026-11-03-accepted-baseline/`；该目录受现有忽略规则保护。

## 修改归属

[逐文件登记](benchmarks/accepted-baseline-inventory-2026-11-03.json)覆盖开始时全部未提交文件：
`keep` 66、`adapt` 1、`test-only` 39、`exclude` 131。

- `keep`：已验收 Rust 生产链、依赖及轻量文档证据；混合文件保留现有 `cfg(test)` 边界。
- `test-only`：属性回归、真实素材入口、来源/候选/逐点诊断与 Node 契约。研究结论不进入生产 owner 决策。
- `adapt`：`package.json` 同时含堆栈和亮度脚本；独立堆栈基线只纳入堆栈脚本，工作区保留全部脚本。
- `exclude`：其他前端/亮度工作、本地 `.agents` 原始诊断、图片及设置。仅从堆栈基线提交排除，不删除文件。

## 已完成的实际验收

冷/热运行均为 Quality Gate `pass / 8 ROI`，输出逐字节一致；9 个 Source_RAW 均贡献像素，
covered owner-zero 为 0。原生裁片和来源证据绑定同一输出 SHA，概览检查仅用于整体结构。
相对同环境顺序对照，耗时下降 33.95% / 31.16%，RAW 解码 85 → 67；fusion 均为 6 次。
峰值 RSS 没有稳定下降，不据此声明内存优化完成。

验收使用 source cache 2.4 GB、native fusion cache 4 GiB；不承诺默认 512 MiB 配置下相同性能。
MTF、boundary 内容不适用与 grouping/low-sharpness degraded 保留；旧严格 RGBA 诊断失败不改写。
完整最佳焦点仍未证明。详见性能优化文档第 29 节。

## 当前工作区集成验证

本轮独立记录当前前端与 Rust 检查，不以旧版验收代替新的执行结果。
原始日志、命令、退出码和耗时见本地 `output/integration/2026-11-03-accepted-baseline/`。

当前已执行：

| 检查 | 结果 |
| --- | --- |
| `cargo fmt -- --check`、`cargo check --locked --offline --lib` | 通过 |
| `cargo test --locked --offline --release --lib` | 1017 passed / 0 failed / 40 ignored，138.38 s |
| 在 `src-tauri` 内使用固定 Rust 1.96.1 执行同一完整测试 | 1017 passed / 0 failed / 40 ignored，114.77 s；测试目标编译 8m 22s |
| TypeScript、零告警 ESLint、Vite build | 通过 |
| 报告质量契约、数据集契约、回归拒绝规则及其他 Node 边界检查 | 通过；报告契约修复后复跑通过 |
| 12 语言翻译同步、运行时复数与语言选择 | 补齐新键后通过；1008 个复数解析、8 个语言选择用例 |
| 手动亮度、自适应亮度、接缝候选合成检查 | 通过；不替代真实接缝识别验收 |
| 原始 9 NEF、冷/热 TIFF、source binding、3 份视觉记录 | 共 15 项 SHA-256 重新核对通过 |
| `npm run tauri -- build --bundles app --ci -- --locked --offline` | macOS 应用包构建通过；Rust 编译 6m 07s，保留 187 条编译告警 |
| 实际桌面导入 → 曝光调整 → 导出 | 通过；独立 NEF 副本，8256×5504 sRGB JPEG，实际打开检查正常；撤销/重做恢复显示值 |

验证期间检测到其他工作对 `seamCandidates.ts` 和亮度评审产物的更新，未覆盖这些文件。
最新前端源码已重新绑定哈希，并复跑类型、lint、翻译及亮度/接缝检查。
收口时两份 `brightness-review.html/jpg` 已被外部工作删除；保留其最新删除状态，不恢复覆盖。
它们开始时的内容仍在 237 文件快照中，最终保护核对记录在本地 `workspace-preservation.json`。

打包后 `seamCandidates.ts` 又有一次并发更新，已补跑该版本的接缝测试、类型、lint 和 Web 构建，
检查期间哈希稳定。已测试的 app 包和更新后的 Web 构建分别记录哈希；前者的单图桌面冒烟不被
扩展为新接缝实现的桌面或真实素材验收。Rust 源码始终与冻结基线相同。

最终结果为 **通过本轮范围，保留非阻断格式债务**；22 个前端/Node 检查命令通过，
两套工具链完整库测试均通过。[机器可读摘要](benchmarks/workspace-integration-2026-11-03.json)
绑定详细记录 SHA-256；所有原始日志保留，初次契约/翻译失败未被覆盖为通过。

桌面验证使用新生成的 `src-tauri/target/release/bundle/macos/RAW Editor.app`。
副本最初继承源文件的 `0700` 权限，文件选择器 Open 不可用；仅将副本改为 `0644` 并刷新后可正常导入。
原始 NEF 的字节与权限不改动；验收副本、编辑导出 JPEG 及应用二进制哈希记录在本地 `desktop-smoke.json`。
实际显示曝光 +0.50，导出文件已通过尺寸/ICC 检查和缩略视觉检查；这不是新增的 1:1 堆栈画质验收。

已定位并修复两处同步遗漏：

1. Node 报告契约和设计文档补入已存在的 `performance` 字段，保持严格顶层字段匹配；不修改生产判据。
2. 为亮度工具的 11 个新键补齐 12 个语言包，仅增加缺失键，保留原有文案和工作区修改。

全库 Prettier 首轮发现 48 个文件的格式告警，涉及既存文档、历史报告、基础代码和其他未提交修改。
CI 将格式检查设为非阻断；本轮记录这一债务，不进行全库自动格式化，不宣称格式检查全绿。

## 后续边界

继续保留 Langyuan/Wenyuan/84 张等跨素材门禁，不能用本次 9 张验收勾选整个 N3–N8。
亮度工具的合成测试通过不等于真实接缝识别通过；自动接缝修复仍不启用。
当前集成检查及单图桌面冒烟不代替所有编辑器操作、其他相机素材和 Windows/Linux 发布验收。
本地 app 构建不代表已公证、签名分发或完整发布 CI 通过；未执行全特性严格 Clippy。

## v34 倍率契约回归核验（2026-11-04）

新增几何契约不替换本文件冻结基线。相同9张NEF、scale=1、默认512MiB/1GiB缓存的新冷/热均QG pass /8 ROI，RAW18/18、fusion6/6、source/native hits113/15。TIFF SHA与本基线完全相同，12个非计时证据部分均与v32一致；旧v32报告与TIFF本体重新核验。wall401.591/376.232s、peak RSS9.707/9.513GB，预算通过。v28基线分支仍指向`504d904ff77781ac13e57f52c13fef8024b6c961`。

阆苑/文苑恢复几何连通但各有覆盖/闭环失败，不加入已验收基线。证据见[倍率复验摘要](benchmarks/cross-station-v34-variable-scale-acceptance.json)与[方案](focus-stack-variable-magnification.md)。

## v35 联合几何与覆盖测量回归（2026-11-04）

旧基线、分支、二进制及报告保持原件。相同9张NEF默认512MiB/1GiB、scale=1冷/热均QG pass /8ROI /8 station owner evidence，RAW18、fusion6、source/native hits113/15；缓存预算与来源证据检查通过。两轮TIFF SHA仍与v28相同，11个非QG证据部分和其他QG判据不变。有效覆盖改为独立原生物理域99.989321%，历史矩形和alpha计数保存在审计中，Composition不变。

冷/热wall532.148/677.405s、峰值RSS8,794,636,288/9,421,586,432B。比v34单次wall明显增加，精确覆盖约2秒不能解释全部差异，尚未完成受控性能归因；本项仅确认画质/来源/缓存回归，不宣称性能无回退。冷/热为独立进程，OS页缓存未清空。视觉证据按TIFF逐字节相同继承。

文苑10张完整原生验收通过；阆苑10张覆盖与PS局部注册修复后QG通过，但closure=no_constraints仍被原生验收拒绝，不加入完整跨素材已验收范围。详情见[最终摘要](benchmarks/cross-station-v35-native-domain-acceptance.json)及[优化文档第36节](focus-stack-performance-optimization.md#36-v35联合闭环与有效来源域2026-11-04)。
