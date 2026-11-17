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
- `build.json`：label、二进制与wrapper SHA、原始提交、来源记录、源码清单和冻结时间。

label 仅允许1–128个字母/数字/`._-`，首字符为字母或数字。已有标签一律拒绝，包括并发重复请求；发布通过同一文件系统内的目录重命名完成，读者看不到半成品。失败和正常可处理的中断清除暂存目录；KILL 后可能留下隐藏暂存目录，不视作已发布版本。文件设为只读，工具层面禁止覆盖；这不阻止文件所有者主动修改文件权限。

任务2使用 `common.verify_binary(label)` 核对二进制及wrapper摘要后运行。任务2的 `run.py`、`matrix.py`、回归预期、RSS/24GiB门、受影响素材矩阵尚未在任务1验收。

## 验证

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/stack-lab -v
```

测试使用隔离临时Git仓库和模拟Cargo进程，不触碰原素材、不做Rust构建。测试需要操作系统允许设置nice与发送进程信号；不要在拒绝setpriority的沙箱里运行后忽略失败。真实基线采用及真实Cargo构建的证据另见 [任务1记录](../../docs/artwork-lanes/1-resource-freeze.md)。
