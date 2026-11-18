# 资源槽与版本冻结（任务1）

开发规则见 [tasks.md §6](../../.kiro/specs/layered-camera-group-focus-stitching/tasks.md)。仅依赖 Python 3 标准库和 Git，构建另需 Cargo。支持 macOS/Linux 的 `flock`、进程组和调度优先级；不提供 Windows 实现。

## 资源槽

```sh
python3 scripts/stack-lab/slot.py status
python3 scripts/stack-lab/slot.py build --lane task-1 --what 'compile' -- cargo check --offline
python3 scripts/stack-lab/slot.py run --lane task-2 --estimate-gib 12 --what '9 NEF' -- executable args
python3 scripts/stack-lab/slot.py exclusive --lane task-5 --what 'first full run' -- executable args
```

- `--` 必填；槽的选项放在它之前，后面的参数原样交给命令。嵌套占槽直接拒绝，防止自锁。
- 所有 worktree 从 Git common directory 找到主仓库，共享 `output/acceptance/.slots/`。`build`、`run` 各一把内核锁；`exclusive` 固定先取 run 再取 build，等待期间可以预留 run，状态标为 `acquiring`。
- 同类请求阻塞等待并显示占用者。build/run 可重叠，记录开始时已有的占用和完整历史时间，供任务2判断整个运行区间的 `contended`。这里不作性能结论，也不实现任务2的完整归因分析。
- build 最多 `CARGO_BUILD_JOBS=3`；run 最多 `RAYON_NUM_THREADS=5`；exclusive 最多8渲染线程、6编译作业。继承的较小正整数保留，过大、无效、0/负值收敛至上限；构建还收敛 CMake 并清除继承的 Make 并发参数。
- 子命令的实际 nice 至少10，设置失败就拒绝启动。环境变量是工具约束，调用者不得在命令参数或子进程中另行放大并发；本工具不是操作系统 CPU 配额隔离器。
- run 必须给出有限正数 `--estimate-gib`，且≤20；未知峰值、预估>20、首次84/98/126张和性能计时使用 exclusive。**24 GiB 是实际RSS验收门，任务1不采样、不据此自动杀进程**；运行摘要及资源门判定在任务2。
- INT/TERM/HUP 中断会终止整个命令进程组，再释放槽。子进程继承锁描述符；监督进程被 KILL 后，原命令存活时仍占槽，status 能显示其记录。KILL 不可能补写退出历史；原命令退出后内核释放锁，旧状态文件不再被视为占用。
- 当前 macOS 执行沙箱拒绝 `setpriority`，需使用允许降低调度优先级的执行环境；工具不会因被拒绝而按普通优先级继续运行。

## 冻结

```sh
python3 scripts/stack-lab/freeze.py --label candidate-r1 --worktree .worktrees/task-1
python3 scripts/stack-lab/freeze.py --label base-2026-11-12 --adopt output/acceptance/base-p1/candidate-release-binary
```

freeze 自动占 build 槽，**从构建/采用到复制、核对、发布全过程持有该槽**。在已有 build/exclusive 槽中直接调用也可，不重复占槽。只允许作为监督进程的直接子命令运行。

新构建使用主仓库 `src-tauri/target`，执行 `cargo test --release --lib --offline --no-run --message-format=json`，从 Cargo 消息精确选出唯一库测试可执行文件，不运行库测试。冻结前后核对提交与构建输入摘要，输入变化就拒绝发布。源码清单包含 Rust、Cargo、build.rs、配置、资源和未忽略的新输入；删除的已跟踪输入记为空摘要。外部工具链、依赖缓存及忽略文件不在此清单中，不宣称覆盖一切外部构建条件。

`--adopt` 要求可执行文件旁有 `build.json`，其中的 SHA 必须匹配，且包含原始完整提交与源码清单。保留原记录的来源，不根据当前 HEAD 猜测旧二进制的提交。旧记录只列 Rust 源码时照实保留范围；`frozen_by_commit` 单独记录冻结时的工作区提交。二进制内容和原有记录是来源依据，不宣称重建可复现。

发布至 `output/acceptance/binaries/<label>/`：

- `candidate-release-binary`：复制后的独立只读可执行文件；
- `native-process-wrapper.py`：执行入口，输出 `NATIVE_PROCESS_USAGE`（RSS统一成bytes）；
- `build.json`：label、二进制与wrapper SHA、原始提交、来源记录、源码清单和冻结时间；`resource_contract`记录实际二进制限额探针是否通过。

label 仅允许1–128个字母/数字/`._-`，首字符为字母或数字。已有标签一律拒绝，包括并发重复请求；发布通过同一文件系统内的目录重命名完成，读者看不到半成品。失败和正常可处理的中断清除暂存目录；KILL 后可能留下隐藏暂存目录，不视作已发布版本。文件设为只读，工具层面禁止覆盖；这不阻止文件所有者主动修改文件权限。

冻结时会在Rayon=1下实际执行`stack_resource_budget_contract`测试；只有全局池与独立预处理池都遵守预算才标为version=1/verified。缺少探针的历史版本可保留冻结，但标为unverified；探针失败则拒绝发布。探针线程数是验证环境，不是后续素材运行的实际线程数。

任务2使用 `common.verify_binary(label)` 核对二进制及wrapper摘要后运行。任务1只验收资源槽与冻结；任务2的运行、矩阵和RSS门说明见下文，真实结果见 [任务2记录](../../docs/artwork-lanes/2-run-matrix.md)。

## 验证

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/stack-lab -v
```

测试使用隔离临时Git仓库和模拟Cargo进程，不触碰原素材、不做Rust构建。测试需要操作系统允许设置nice与发送进程信号；不要在拒绝setpriority的沙箱里运行后忽略失败。真实基线采用及真实Cargo构建的证据另见 [任务1记录](../../docs/artwork-lanes/1-resource-freeze.md)。

## 单次素材运行（任务2）

```sh
python3 scripts/stack-lab/run.py --binary task-4-capped-r2 --lane task-2 --dataset sleeping --subset sleeping-9-6479 --label r1
python3 scripts/stack-lab/run.py --binary task-4-capped-r2 --lane task-2 --dataset lanting --framed --label r1
```

- 运行当前checkout的`tests/focus-stack-regression.mjs`，只用冻结的二进制与wrapper；每次输出到新目录，已有目录拒绝复用。没有更新或重设已验收基线的选项。
- 输入与参考图只读，前后核对摘要；执行记录保存来源清单、冻结版本、输入/runner摘要、槽、环境、时间和退出码。来源清单按Node runner相同的全局排序→子集→倒序规则构造。
- 只有二进制限额合同已验证，且已有历史峰值的9张、两套10张、兰亭，以及0.1祁序/枫鹰，才自动用run槽；未知范围、首次全量、预估>20GiB或`--exclusive`占独占槽。给未知范围手动提供较小估计不会取消独占要求。旧冻结版本缺少限额验证时一律exclusive，也拒绝将其手动嵌入共享run槽。
- `execution.json`记录过程，`analysis.json`记录QG、完整判据状态、来源计数、闭环、输出TIFF SHA、默认缓存预算/计数、内存与争用证据。多份报告中只接受唯一一份有实际QG判据的报告；不忽略有测量的冲突报告。
- `ps`每0.5秒采样runner进程树RSS之和（KiB→bytes）；超过24GiB立刻终止命令进程组。采样失败也停止运行。最终峰值取采样值、原生`getrusage`及Stack_Report RSS的最大值；要求采样和原生usage都有证据。超过门限、证据缺失或监控错误都不能通过内存门。
- 单次返回0仅表示原生质量/来源/输出/内存全部通过，或显式预览成功（预览状态为`rendered_without_quality_acceptance`，不计入原生验收）。拒绝、资源失败、分析错误返回非零。已启动的runner取消时写入中断状态并停止其进程组。
- `contended`核对完整运行区间与build历史的重叠，包括运行中才开始又结束的编译；不是只看启动瞬间。独占运行不会把自身的build占用误记为争用。
- 兰亭`--framed`只适用于完整45张、scale=1：运行目录内再隔离一层`lanting/`，纸面审计写到其独立父目录；执行严格及regression-only保护，任一失败都拒绝。图像解码、取景、纸面抽样和保护阈值沿用原基线。
- 流式 mosaic 的 owner/coverage 诊断可显式使用`--streaming-diagnostics`；它只设置`RAW_EDITOR_STREAMING_MOSAIC_DIAGNOSTICS_DIR`并把导出写到该运行目录的`diagnostics/`，不启用 framed-artwork、不改变像素或质量门。执行记录保存开关和目录；该选项只用于 test-only 诊断。
- 所有输出限制在`output/acceptance/<lane>/`，全局空闲<40GiB或lane已有输出≥20GiB时拒绝新运行。超限/取消的残留输出是失败证据，不能作为已交付图像。

`--env RAW_EDITOR_KEY=VALUE`用于明确的非合同实验开关；内部输出/报告/内存/阈值等保护项不能覆盖。自定义数据根目录仍通过进程环境`RAW_EDITOR_FOCUS_STACK_DATA_ROOT`指定。

## 回归矩阵（任务2）

```sh
python3 scripts/stack-lab/matrix.py --tier quick --binary base-2026-11-12 --label r1
python3 scripts/stack-lab/matrix.py --tier medium --binary base-2026-11-12 --label r1
python3 scripts/stack-lab/matrix.py --tier extended --binary base-2026-11-12 --label r1 --dry-run
python3 scripts/stack-lab/matrix.py --tier full --binary base-2026-11-12 --label r1 --dry-run
```

版本化预期在当前checkout的`tests/acceptance/regression-matrix.json`；冻结二进制与资源槽仍由所有worktree共享。四层是独立选择，不自动累计：

| 层级 | 内容 |
| --- | --- |
| quick | build槽内依次执行fmt、cargo check、冻结库测试、4项Node合同检查；Rust库测试3线程，Rayon3 |
| medium | 9张冷/热、阆苑10、文苑10、兰亭45取景与严格保护，共5次独立运行 |
| extended | 祁序联合、枫鹰0.1；保持已知拒绝特征，匹配只表示回归一致，`quality_accepted`仍为false |
| full | 当前已达D3及以上且有全量基线的素材；目前只有兰亭45，不把子集通过外推到全量，不含暂缓PSD |

每行核对TIFF摘要、QG及各判据、闭环、来源计数、预算和已知失败特征。来源变动、二进制变动和内存失败不能通过“已知QG失败”豁免。冷/热各起独立进程且各用独立alignment cache，不清OS页缓存。

每个不符项自动通过exclusive重跑一次；两次分析与差异都保留，仍不符则矩阵返回1。独占重跑恢复一致时标`pass_after_exclusive_retry`，不抹掉首轮不符。矩阵取消停止当前任务，不启动后续或自动重跑。

输出为`output/acceptance/<lane>/matrix-<label>-<tier>/matrix.json`及每个case的attempt目录；保存完整预期快照和预期文件SHA。可用`--case <id>`限定层内case，或`--matrix <copy.json>`验证独立预期；工具不会写回预期文件。复现负向验证时只改副本中一项TIFF SHA，预期是两轮实际运行后返回1，而不是修改正式基线。

限额修复与旧版本兼容范围见 [任务4记录](../../docs/artwork-lanes/4-two-slots.md)。`base-2026-11-12`仍是像素比较基准；其旧预处理池不受全局Rayon上限约束，所以从本版本起实际运行自动独占。共享两槽场景使用经过探针与真实回归的新冻结版本。
