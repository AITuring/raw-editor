# 任务1：资源槽与版本冻结

- 日期：2026-11-17。
- 角色：本次由同一开发者实现并承担 L0 集成检查，未启动并行 agent。
- 起点：`v1 b845bc9b`；隔离工作区 `.worktrees/task-1`，分支 `lane/task-1/resource-freeze`。
- 修改范围：`scripts/stack-lab/{common,slot,freeze}.py`、工具说明、隔离工具测试和本任务证据；未改 Rust/前端处理逻辑。
- 任务2的 `run.py`、`lanting_audit.py` 草稿及 `.vscode/settings.json` 原有修改保留，不纳入本任务提交。

## 实现

- build/run 独立内核锁、exclusive 固定顺序占两槽；等待显示占用者，status 根据真实锁判断占用，JSON 原子写入。
- build最多3作业、run最多5渲染线程、exclusive最多6作业/8渲染线程；继承环境变量只能降低上限。nice实际至少10，设置失败则拒绝运行。
- 预估峰值未知或>20GiB要求exclusive；24GiB实际验收门与RSS采样留给任务2。
- INT/TERM/HUP 清理命令进程组及部分获取的锁；监督进程被KILL时，子命令继承内核锁，避免原命令仍运行却被另一个任务进入。
- 冻结从编译/采用直至复制校验和发布，全程持有build槽；label不可复用且禁止路径逃逸；只发布完整目录。
- 新构建记录输入清单并在构建前后核对；采用旧产物要求相邻来源记录并核对SHA，保留原始提交及源码清单，不误记为当前HEAD。
- 冻结二进制、wrapper和元数据只读；运行前校验二进制与wrapper的摘要。

## 验证

隔离进程测试命令：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/stack-lab -v
```

17项测试全部通过（最终一次9.799秒）。涵盖同类build/run排队、占用者可见、build/run重叠与初始争用记录、exclusive双槽及等待取消、线程上限与nice、20GiB调度边界、非零退出码、启动失败释放、INT/TERM/HUP、监督进程KILL后的子进程占锁、拒绝嵌套、冻结中断清理、构建失败/构建中源变化拒绝发布、并发重复label、损坏来源元数据拒绝、原始来源保留及二进制/wrapper篡改检测。

本机macOS执行沙箱拒绝setpriority。已实测沙箱内工具返回127且没有执行目标命令；17项进程测试及真实冻结在允许setpriority的环境运行，实际nice=10。测试不跳过此要求。

### 真实基础版本

- 来源：`output/acceptance/base-p1/candidate-release-binary`，相邻来源记录 `base-p1/build.json`。
- 冻结：`output/acceptance/binaries/base-2026-11-12/`。
- SHA-256：`39bfe6f2aecc8a7ebc154efb94ba4b1f1a26257c825a676a0e54e8cde9cef03b`，与base-p1摘要完全一致。
- 原始提交：`c0c1e5a9037fb4f8a8f8b983802848feb667b9ba`；523项源码清单与原记录完全一致。
- 在run槽以5线程、nice=10运行冻结wrapper的`--list`：退出0，列出1137 tests / 0 benchmarks。此操作仅验证可执行入口，没有执行1137项测试。
- 再次使用相同label返回1，原build.json摘要不变。
- 机器本地产物：`output/acceptance/task-1/base-validation.json`、`base-binary-list.log`。

### 真实编译路径

实际运行：

```sh
python3 scripts/stack-lab/freeze.py --label task-1-build-2026-11-17 --lane task-1 --worktree /Users/dp/Desktop/raw-editor
```

- 使用共享target、build槽、最多3个Cargo作业；实际Cargo/rustc进程nice=10。
- release离线构建成功，Cargo报告耗时7分29秒，冻结目录 `output/acceptance/binaries/task-1-build-2026-11-17/`。
- 提交 `b845bc9b73b6ed09f7e47aae7c1e5a9d06b8d0f0`，719项输入清单；构建前后清单及提交一致，源代码无未提交变更。
- 二进制SHA `b20d19f6919dd53dff7ead03233e3f288522d62a02afcbf1b174d248bcc20469`；此新构建只验证工具，不替换已验收base的二进制或图像基线。
- 新冻结wrapper的`--list`与3个定向Rust测试通过：规范输出编码、非对称门槛下规范方向与导入顺序无关、整簇修复预检拒绝。未重跑完整Rust库测试。
- 本地产物：`output/acceptance/task-1/compiled-validation.json`、`compiled-smoke-0..3.log`，构建消息保留在冻结目录的`cargo-build.jsonl`。

机器摘要：[task-1-resource-freeze.json](../benchmarks/task-1-resource-freeze.json)。

## 验收边界与后续

任务1不重跑图像中等层、不新增画质或性能结论、不变更任何素材基线；中等层自动矩阵属于依赖任务1的任务2。现有基础版本的内容身份通过二进制SHA与来源记录保持。

任务2继续实现run/matrix、RSS/24GiB门、完整contended归因和矩阵预期；任务3补lane/诊断卡及复看队列模板；任务4用9张NEF验证线程数与两槽演练对输出的影响。任务1的工具测试不代替任务4的真实像素验收。
