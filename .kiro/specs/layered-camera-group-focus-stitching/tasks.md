# 堆栈模块开发任务（新基线）

## 使用规则

本文件是后续堆栈模块开发的唯一执行计划。旧版任务编号、阶段表、逐轮实验记录、临时报告路径和历史
分支状态不再作为任务状态，也不作为验收依据。历史报告可以留在 Git 历史和 `.agents/` 归档中，但不能
直接勾选本文件中的任务。

当前基线是 `v1`：2026-10-28 将远端 `d58ff8aa` 与本地快照 `a4f27d09` 合并。
本次保留的候选修改仍须经过本文件的 N0 分流和验证，合并、编译和回归测试不代表真实素材验收完成。
合并前的本地执行记录保存在 `docs/focus-stack-local-work-history.md`，仅供历史追溯，不改变下列任务状态。

`.worktrees/layered-focus-cleanup` 的模块化分支不是当前基线；其中的行数和完成状态不能直接写入本文件，
只有经过 N0 的逐项迁移、编译和回归后才能合并进来。

执行纪律：

- 先记录证据，再改变行为；所有行为变更都要有 baseline、输入 digest、输出 digest 和可复现命令。
- 每个阶段只解决一个职责层，阶段结束后再进入下一层；不得把测试迁移、算法调参和输出改造混在一个提交中。
- 生产代码、测试支持代码和 test-only oracle 都必须保持职责清晰；单个文件最长 1000 行。
- 原有门槛保持不变，尤其是 station relation 的尺度、空间支持、方向、双向匹配和闭环门槛；没有独立证据
  不得放宽阈值、强制连接机位或把外部变换写入生产关系图。
- Quality Gate 在真实素材连续通过前保持 record-only；导出成功、结构测试通过或 `not_run` 都不能代替门禁通过。

## 当前代码基线

### 需要纳入 N0 审计的生产范围

- 顶层编排：`src-tauri/src/panorama_stitching.rs`。
- 拼接核心：`src-tauri/src/panorama_utils/stitching.rs`、`mosaic.rs`、`processing.rs`。
- 堆栈阶段：`stack_pipeline/{focus_fuser,quality_gate_runner,virtual_tile,report,tone,quality_gate,
  intra_station,degradation,residual_warp,compositor}.rs`。
- 测试与属性支持：`panorama_stitching_*_tests.rs`、`panorama_virtual_tile_tests.rs`、
  `stack_pipeline/{properties,test_support}.rs`。

当前工作区这些文件仍有超过 1000 行的生产或测试实现。N1/N2 必须重新测量实际文件，而不是引用其他
分支的行数结论。

### 未提交修改的分流对象

N0 要逐项登记以下候选修改的来源、依赖、测试和生产状态：

- 可保留候选：canonical station relation gate、decoded/preflight、cache/proxy 坐标契约、确定性辅助、
  Quality Gate rejection/effective-pixel 记录和 BigTIFF 路由；每项仍须通过 N0 的 clean regression。
- 需要重新验证后才能接入生产：`feature_provider`、`focus_pyramid`、`proxy`、`station_seam`、
  `tile_source`、Residual_Warp 接线和新的 compositor 路径；owner smoothness、FlatArtwork、direct
  matcher 和 global/prior-free bridge 先隔离到实验路径。
- 只允许 test-only 或研究隔离：`panorama_structural_oracle.rs`、`owner_quality_oracle.rs`、
  `final_world_owner_oracle.rs` 以及任何 reduced score、ROI reservation、QG replay、外部 Source_RAW
  关系或未经 holdout 的 pose/warp oracle。
- `src/components/modals/BatchGeometryModal.tsx` 属于无关 UI 修改，不纳入堆栈基线。
- 原始 command log、Stack Report、preview、RAW 和绝对临时目录不进入源码提交；只保留脱敏后的 schema、
  计数和可复现命令。

### 必须保持的产品契约

- Rust 是默认生产链，负责 RAW、全分辨率 IO、几何验证、融合、输出、取消、内存和确定性；Python/OpenCV
  只能作为可选代理 provider 或离线对照，失败必须回退 Rust 并写入报告。
- Source_RAW 的 digest、来源 provenance、Coverage_Mask、Ownership_Map 和透明区域语义必须贯穿渲染、
  诊断和输出；不得用 Photoshop 结果、文件名顺序或外部变换替代生产证据。
- station relation 必须经过同一 typed canonical view 和最终共享 gate；proposal 只能产生搜索种子，
  不能直接升级为 accepted relation。
- owner 证据必须来自当前运行的最终坐标和 Source_RAW 采样；test-only reduced oracle 不能替代正式 owner。
- 没有足够双向闭环证据时 Residual_Warp 保持 identity，并把原因写入报告。

### 当前验收基线

- 兰亭 45 张基准已恢复到 45/45 输入完整参与渲染。N1-2a 后的 1:1 回归输出 `33895×13888`，成功写出
  `/private/tmp/raw-editor-lanting-coordinates-full.tiff` 和同尺寸 app JPEG，预览 `1600×656`；日志包含 45 条
  source streaming 记录，整图及关键裁片笔画可辨，参考文件保持只读。
- 兰亭当前通过 source-scan fallback 走 `legacy_single_layer_mosaic`，避免把移动扫描误判成多机位 station 后
  在 LayeredVirtualTile 中排除帧。Stack_Report 仍会保留 grouping isolated 的诊断事实；这不等于 geometry/quality
  gate 已通过，后续 N4 需要单独解决 station grouping 与画布裁切语义。
- N1-2b 后的 10% 保护性回归仍为 45/45 source streaming，画布 `3388×1389`，预览 `1200×492`；报告仍保留
  `DSC08871.jpg` native refinement 失败、43 个 isolated 和 `quality_gate=insufficient_evidence`，因此只确认合成
  行为没有回退，不把本次 preview-only 运行标记为 Quality Gate 或正式输出通过。
- 当前恢复保住了“45 张完整且处处清晰”的行为门，但输出画布尚未与参考 Photoshop 成片 `31065×11007` 对齐；
  Stack_Report 的 `output.written=false` 是因为验收测试在报告生成后另行写出 TIFF/app JPEG，不能据此判断测试文件未写出。
- 数据契约覆盖 8 个数据集并通过：兰亭 45、阆苑女仙 84、文苑图 126、张好好2 244、睡觉 45、枫鹰雉鸡
  63，以及头部 17、底部 24。
- Rust 完整 release 库测试在 N1-2b 后为 `818 passed, 0 failed, 30 ignored`。两个属性测试的回归根因是已有 station anchor
  被重新按清晰度排序；现在已改为“已有记录锚点优先、无记录时才按清晰度选锚点”，属性测试和完整套件均通过。
- Langyuan/Wenyuan 的 owner、effective-pixel、boundary、tone 和 residual evidence 仍不能视为通过；不能用兰亭
  预览恢复代替完整门禁。owner smoothness/hard terminal、ROI reservation、QG replay、Option B、spatial tone、
  直接 Source_RAW bridge 和 structural oracle 继续保持 test-only/research 隔离。

N3 继续用固定 manifest 重新生成每套素材的解码、机位图、Quality Gate 和输出证据；兰亭的 45/45 行为门是进入
N1 后每个职责层的回归保护条件，其他数据集的历史结果不能直接标记为当前通过。

## N0：冻结新基线并分流未提交修改

- [ ] 保存当前 `git status`、tracked diff、untracked 清单和忽略规则摘要；不 reset、stash、clean 或覆盖
  工作区已有代码。
- [x] 建立 `tests/acceptance/focus-stack-datasets.json` 和 `focus-stack-datasets:check`；登记兰亭、阆苑女仙、
  文苑图、张好好2、睡觉、枫鹰雉鸡及其头部/底部输入，排除派生图；兰亭 45 张输入的参考成片路径、
  SHA-256、尺寸和文件大小已冻结。
- [x] 恢复兰亭 45 张完整成片行为：45/45 输入进入 streaming compositor，1:1 回归成功写出 TIFF/app JPEG，
  整图与左中右裁片保持清晰；参考成片未覆盖。输出尺寸/裁切对齐仍作为后续独立门禁。
- [ ] 为每个修改建立 `keep`、`adapt`、`test-only`、`exclude` 四类登记，注明生产入口、测试入口和证据。
- [ ] 清理源码树中的 raw/preview/命令报告引用；历史 artifact 只保留在本地归档，不进入生产提交。
- [ ] 在不改变行为的前提下完成 `cargo fmt --check`、`cargo check --lib`、定向 Rust 测试、Node 契约、
  `npm run typecheck`、`npm run build` 和 `git diff --check`。
- [ ] 将筛选后的当前代码形成一个可回滚基线提交；此提交之后才开始 N1 的拆分工作。

**N0 通过条件**：每个未提交文件都有归属；没有未登记的实验 oracle 或大文件 artifact；基线能编译，且
所有失败都能归因到当前代码而不是混杂的历史工作区。兰亭参考文件不得被测试覆盖，且每次基线/行为变更
后都必须重新确认其 SHA-256、尺寸和文件大小未改变。

**N0 当前状态**：兰亭保护性回归已恢复。当前 45 张全部进入顺序并完成 streaming compositor，1:1
成片范围为 `33895×13888`，预览 `1600×656`，并已在 `/private/tmp` 写出 TIFF/app JPEG，整图和关键裁片保持
可读；参考成片仍为只读且 SHA-256、尺寸、文件大小未改动。输出尺寸/裁切与参考不同，Quality Gate 和最终发布
仍未宣称通过；N1 可以在保持该行为门的前提下继续，尺寸对齐另列为后续任务。

## N1：按职责拆分生产代码

拆分只移动代码和收敛依赖，不改变算法阈值、坐标含义、回退行为或输出格式。推荐顺序：

1. 共享类型、错误、报告和坐标转换。
2. 纯 geometry/focus helper、采样、mask 和颜色计算。
3. feature/matching、focus grouping、station solver、topology 和 closure。
4. focus warp、residual warp、ownership、seam 和 compositor。
5. virtual-tile store、tile source、BigTIFF/output adapter。
6. 顶层 prepare、执行流程、取消、缓存和最终发布。

- [x] N1-1 共享类型迁移：新增 `panorama_utils/shared.rs`，承载 BRIEF descriptor、`ImageInfo`、
  `MatchInfo` 和 focus warp 合同；facade 保留兼容重导出，低层模块改为直接依赖 shared，编译和定向测试通过。
- [x] N1-2a 纯坐标契约迁移：新增 `panorama_utils/coordinates.rs`，承载 `Projection`、投影/逆投影、
  `map_target_to_source`、`output_bounds`、`pixel_aligned_canvas` 和 `output_canvas_dimensions`；
  `stitching.rs` 保留兼容重导出，3 个新坐标单测、旧 facade 单测、property 54、数据集契约、兰亭 10% 回归和
  1:1 写盘回归通过。
- [x] N1-2b 纯 geometry/focus helper 迁移：新增 `panorama_utils/render_geometry.rs` 和
  `panorama_utils/focus_geometry.rs`，分别承载 transformed image/source bounds、focus output/render bounds、
  mapped frame 与 shifted-mosaic 判定；模块均低于 1000 行，保留 `stitching.rs` 兼容重导出。新增 geometry
  单测、shifted-mosaic 定向测试、virtual-tile/focus-analysis 回归、数据集契约、兰亭 10% 回归和完整 release
  库测试通过，行为和诊断阈值未改变；随后用同一 release 二进制完成兰亭 1:1 写盘回归，输出
  `33895×13888`，成功写出 `/private/tmp/raw-editor-lanting-focus-geometry-full.tiff`、同尺寸 app JPEG
  和预览，45/45 source streaming 保持不变。
- [ ] `panorama_stitching.rs` 只保留稳定入口、阶段编排、常量和重导出。
- [ ] `stitching.rs`、`mosaic.rs`、`processing.rs` 和 `stack_pipeline` 生产模块拆成单一职责 sibling
  模块；facade 不承载算法实现。
- [ ] 所有生产文件不超过 1000 行；模块 import 方向通过源码检查，禁止测试支持反向依赖生产。
- [ ] 新建并维护 `docs/focus-stack-architecture.md`，只描述当前实际模块树和依赖方向，不复制其他 worktree
  的完成数字。
- [ ] 每完成一个职责层，运行该层定向测试并记录迁移前后输出等价性。

**N1 通过条件**：生产模块树可由架构文档生成；入口、缓存、solver、compositor 和 output 的边界清晰；
结构检查、编译和现有行为回归全部通过。

## N2：拆分测试支持和研究 oracle

- [ ] 先拆 `stack_pipeline/test_support.rs` 和共享 fixture，再拆 focus ownership、registration/topology、
  alignment、Virtual_Tile、acceptance 和 interpolation 测试。
- [ ] `properties.rs` 按领域拆分；每个 property 保留原名称、过滤路径、生成器语义和失败样本。
- [ ] `panorama_structural_oracle.rs`、`owner_quality_oracle.rs`、`final_world_owner_oracle.rs` 拆成不超过
  1000 行的 test-only 模块，禁止从生产模块 import。
- [ ] 将 `tests/focus-stack-quality-contract.mjs` 中的 production invariant 与 research/test-only oracle
  断言分开；release contract 不再依赖已停止的 owner、tone 或 structural research。
- [ ] `stack-pair-evidence:check` 只作为 research harness，不能成为 release gate，除非 N3 重新定义并验收它。
- [ ] 每次迁移先跑定向测试，再跑完整 `cargo test --lib`；测试迁移提交不得同时改变生产逻辑。

**N2 通过条件**：测试支持和 oracle 均不超过 1000 行，失败能定位到一个领域，测试数量和过滤路径可追踪，
生产构建不携带研究实现。

## N3：建立新的观测与真实素材验收入口

- [x] 固定 `tests/acceptance/focus-stack-datasets.json` 数据集 manifest、输入扩展名/排除规则和兰亭参考
  成片；使用 `RAW_EDITOR_FOCUS_STACK_DATA_ROOT=/Users/dp/Downloads/书画拼图 npm run
  focus-stack-datasets:check -- --require-data --hash-inputs` 校验数量、输入 digest 和参考文件。
- [ ] 在 manifest 基础上固定 Langyuan-10、Wenyuan-10 和 84 张 NEF 的解码状态、network=0、输出目录和
  运行参数；不把旧临时路径或旧报告 hash 写入任务状态。
- [ ] 每个职责层完成后先运行上述数据契约检查；涉及成片行为时，必须用兰亭 45 张真实输入运行
  `real_focus_stack_fixture_from_env`，将输出和 `RAW_EDITOR_STACK_REPORT_DIR` 都写入 `/private/tmp`，并
  检查 45/45 输入、整图与左中右清晰度裁剪、Coverage/Ownership 证据和参考成片对照；禁止覆盖
  `/Users/dp/Downloads/书画拼图/兰亭/DSC08854_FocusStack.jpg`。当前恢复路径仍允许 preview-only，
  但最终发布验收必须另行确认 `output.written` 和实际输出文件。
- [ ] 统一 Stack_Report：input、proxy、grouping、station relation、topology、closure、fusion、owner、
  residual warp、tone、seam、composition、quality gate、output、resources、degradation 和 provenance。
- [ ] 所有诊断使用最终画布坐标，同时保存 source/native/world 坐标和转换矩阵；报告记录失败阶段而不是只写
  一个最终 verdict。
- [ ] 先跑 10 张样本确认报告链路和证据完整，再跑 84 张确认机位图、资源和停止原因；旧验收结果只作为待复核
  的风险提示，不直接标记通过或失败。

**N3 通过条件**：同一 manifest 重跑可复现报告 schema、输入 digest 和输出 digest；真实运行不会因缺字段、
坐标混用或旧 artifact 误读而改变结论。

## N4：先解决 station geometry，再进入画质

- [ ] 核对 source/native/world 坐标、crop、proxy scale、先验布局和机位成员；用 source member 标识，不用
  文件顺序或动态 station 编号作为关系证据。
- [ ] 让 feature/matching、Virtual_Tile、canonical relation gate、prior-free search seed、polish 和
  topology 使用同一候选生命周期：proposal → measurement → shared gate → accepted/rejected。
- [ ] 记录 `no overlap`、support、bidirectional、grid、scale、hull、orientation 和 closure 各阶段，禁止
  直接放宽既有门槛或强制 bridge。
- [ ] 机位图达到目标连通性后，再启用 station pose、intra-station registration 和 residual warp；证据不足
  时保持 identity。

**N4 通过条件**：每套真实素材的机位图可解释、闭环进入可测量状态、独立 holdout 与 production relation
  一致，并且通过的关系不会因线程数或候选遍历顺序变化。

## N5：处理 focus ownership 和融合证据

- [ ] 以当前运行的 registration、最终 world、Coverage 和 Source_RAW 采样生成 owner evidence；记录 winner、
  runner-up、run-length、重复测量方差、ROI 数和 provenance。
- [ ] owner coverage 不能以牺牲 effective pixel、gradient/noise、boundary、ROI 数或 Source_RAW fidelity
  换取；smoothness、hard terminal、ROI reservation 和 content takeover 研究不得自动进入生产。
- [ ] focus warp、Residual_Warp、mask、透明区域和单一 owner 写入保持可追溯；任何 fallback 都写入报告。
- [ ] 只在真实关系和闭环证据稳定后评估跨站融合，先保留原有安全回退。

**N5 通过条件**：owner 结论能回放到最终像素和 Source_RAW，holdout 不劣化，Coverage/Ownership/provenance
  一致，且不依赖 test-only oracle。

## N6：seam、tone、compositor 和大图输出

- [ ] 机位间 seam 只在双方 Coverage 有效且有足够重叠时求解；GraphCut、DP 或中线 fallback 记录 solver、
  代价、超时和机位对，不把无 Coverage 像素纳入接缝搜索。
- [ ] Tone_Harmonizer 使用可解释的 shared-world evidence；空间 tone、owner-fidelity 不可行方案保持研究
  隔离，不以模糊边界掩盖真实色差。
- [ ] TileSource/TileCompositor 按需读取全分辨率块，验证 tile 与 whole-frame 的像素、mask、ownership、
  alpha 和 provenance 等价；单块失败不得发布部分结果。
- [ ] BigTIFF/Pyramidal TIFF 写出支持 tile、overview、位深、ICC、alpha、取消、fsync 和原子 rename；输出
  metadata 和编码失败写入报告。

**N6 通过条件**：Langyuan/Wenyuan 的 seam、tone、边界和输出证据可定位；tile 顺序、缓存命中和线程数不改变
  结果 digest；失败不会残留半成品。

## N7：Quality Gate 与发布边界

- [ ] 先让 Quality Gate 完整记录所有判据和证据；几何断连、closure 不可靠、owner shortfall、tone degraded、
  boundary 超限或证据不足时明确拒绝发布。
- [ ] 只有 Langyuan-10、Wenyuan-10 和 84 张在连续可复现实跑中满足几何、融合、owner、tone、boundary、
  output 和资源条件后，才切换 blocking mode。
- [ ] 发布只接受通过 gate 的最终文件；报告、诊断和失败原因可以保留，但不得把 preview、`not_run` 或旧
  report 复制成最终产物。

**N7 通过条件**：Quality Gate 的 verdict、output.written、报告和文件系统状态一致，失败路径可重复，且没有
  通过修改阈值或绕过证据获得“通过”。

## N8：确定性、资源和可选 provider

- [ ] 对 1/2/8/16 线程、重复运行、取消、内存超限、tile/whole-frame 等价和 provider fallback 做回归。
- [ ] Rust provider 保持默认；可选 Python/OpenCV 只在固定 proxy fixture 上比较匹配率、重投影、seam、tone、
  owner、RSS、耗时和输出 digest，未证明收益不得切换默认 provider。
- [ ] 统一 Rayon/OpenCV 预算、缓存 key、显式拷贝计数、RSS 采样和临时文件生命周期。
- [ ] 输出、报告排序、浮点归约、RANSAC seed、tile 顺序和失败原因保持确定性。

**N8 通过条件**：资源边界和确定性报告可自动比较，provider 失败有安全回退，长任务不会产生不可控的重复
  大数组或残留临时文件。

## N9：最后再扩展算法

- [ ] 真实门禁稳定后，再评估 FlatArtwork 的受限 similarity、轻微曲率的分段 homography、受约束非刚性模型
  和严格裁切/透明/边缘填充策略。
- [ ] 每个新模型必须有独立 baseline、holdout、失败回退、资源预算和开关；不得直接覆盖稳定生产路径。
- [ ] 通用 `processing.rs` 在业务职责稳定后按采样、滤波、颜色和几何 primitive 拆分，并保持公共契约不变。

## 阶段完成检查单

每个 N0–N9 阶段完成时必须同时提供：

1. 代码差异和职责边界说明。
2. 定向测试、完整回归和源码契约结果。
3. 若涉及真实素材，提供 manifest、digest、报告、输出 digest、资源和失败阶段。
4. 更新架构文档和本文件当前状态；只更新当前基线，不追加逐轮历史流水账。

最终完成的定义是：生产文件和测试支持文件均不超过 1000 行，模块依赖方向稳定，真实素材验收可复现，
Quality Gate 与发布边界一致，且所有未验证研究代码都留在 test-only 或独立实验路径。
