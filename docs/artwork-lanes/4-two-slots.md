# 任务4：线程上限核对与两槽演练

- 日期：2026-11-17；基础：`v1 6469b568`。
- 隔离工作区：`.worktrees/task-4`；分支：`lane/task-4/two-slots`。
- 执行与L0：本次由同一执行者完成，不启动并行agent。
- 范围：真实编译/冻结、两次9张运行与排队；发现并修复预处理池绕过全局线程预算的问题。质量阈值和素材像素基线保持。
- 状态：L0接受任务4范围；r1发现的独立池6线程缺口已修复，r2真实两槽、快速层及中等层5个范围全部通过。第0批任务1–4验收完成。

## r1历史输入与命令（修复前）

两次运行均为`sleeping / sleeping-9-6479`，即DSC_6479–6487的9张NEF，scale=1；原素材只读。运行使用`base-2026-11-12`，二进制SHA `39bfe6f2aecc8a7ebc154efb94ba4b1f1a26257c825a676a0e54e8cde9cef03b`，原始构建提交`c0c1e5a9037fb4f8a8f8b983802848feb667b9ba`。

以下是6469b568版本工具执行的r1历史命令；当前工具对旧基线会强制exclusive，不能再用这些旧命令验证共享两槽。r1当时前两个同时执行，第三个等待run槽。

```sh
RAYON_NUM_THREADS=5 python3 scripts/stack-lab/run.py --binary base-2026-11-12 --lane task-4 --dataset sleeping --subset sleeping-9-6479 --label primary-r1
CARGO_BUILD_JOBS=3 RAYON_NUM_THREADS=3 python3 scripts/stack-lab/freeze.py --label task-4-build-r1 --lane task-4 --worktree /Users/dp/Desktop/raw-editor/.worktrees/task-4
RAYON_NUM_THREADS=5 python3 scripts/stack-lab/run.py --binary base-2026-11-12 --lane task-4 --dataset sleeping --subset sleeping-9-6479 --label queued-r1
```

实际调度/验证脚本保存在`output/acceptance/task-4/rehearse.py`；它等待首个execution、build占槽和真实rustc/原生进程出现，再捕获现场。原生运行不读共享target中的可变可执行文件，新的冻结产物只用于本次构建流程验证，不替换已验收基线。

## 现场与隔离证据

- `three-request-snapshot.json`捕获真实rustc和原生`candidate-release-binary`同时存活、两个槽的占用者、进程树nice及排队请求PID。
- `three-request-status.txt`保存当时的slot status；`queued.driver.log`包含`waiting for run`和首个占用者。
- 捕获时第二个请求仍存活，但它的输出目录尚未创建；没有用sleep占槽模拟真实编译或原生处理。
- 输出分别为`output/acceptance/task-4/sleeping-9-6479-primary-r1/`、`sleeping-9-6479-queued-r1/`，alignment cache、Stack_Report、日志和TIFF均按各自目录隔离。
- 完成后按`.slots/history.jsonl`核对第二个请求取得run槽的时间不早于首个释放槽；核对实际build重叠区间与每次`contended`一致。

## r1发现与修复

r1两次TIFF均等于v28、全局Rayon均为5、QG/资源门通过，但原生日志实际打印`Preparing images with 6 bounded worker(s)`。原调度脚本只验证全局Rayon，不能据其pass宣称所有线程预算通过；r1原始报告完整保留。

原因：`scalable_preparation_worker_count`按硬件CPU/内存单独建立线程池，没有收敛到当前Rayon池预算。修复后取两者较小值。新增Rust池预算测试和可执行限额合同探针；冻结时验证实际二进制的探针，旧冻结版本缺失验证标记时由run.py自动改走exclusive，也拒绝把旧版本手动塞进共享run槽。

修复涉及实际调度行为，必须重新验证两槽、9张输出，并完成受影响中等层与快速层；不能凭r1像素相同直接收口。

## r2修复版本与复验命令

原生调度修复提交`e68d24dda7a850d2348f237d4a713933559e9702`。先构建`task-4-capped-r2`，实际限额探针在Rayon=1时报告全局池1、预处理池1并通过；再用该冻结副本执行两次9张运行。r2调度脚本与现场位于`output/acceptance/task-4/r2/`。

```sh
CARGO_BUILD_JOBS=3 RAYON_NUM_THREADS=3 python3 scripts/stack-lab/freeze.py --label task-4-capped-r2 --lane task-4 --worktree /Users/dp/Desktop/raw-editor/.worktrees/task-4
RAYON_NUM_THREADS=5 python3 scripts/stack-lab/run.py --binary task-4-capped-r2 --lane task-4 --dataset sleeping --subset sleeping-9-6479 --label primary-r2 --out /Users/dp/Desktop/raw-editor/output/acceptance/task-4/r2/sleeping-9-6479-primary-r2
# 调度脚本只更新隔离worktree的run_control.rs修改时间，内容SHA不变，让Cargo确实重编译同一源码。
CARGO_BUILD_JOBS=3 RAYON_NUM_THREADS=3 python3 scripts/stack-lab/freeze.py --label task-4-load-r2 --lane task-4 --worktree /Users/dp/Desktop/raw-editor/.worktrees/task-4
RAYON_NUM_THREADS=5 python3 scripts/stack-lab/run.py --binary task-4-capped-r2 --lane task-4 --dataset sleeping --subset sleeping-9-6479 --label queued-r2 --out /Users/dp/Desktop/raw-editor/output/acceptance/task-4/r2/sleeping-9-6479-queued-r2
```

r2额外逐份检查原生日志中的`Preparing images with ... bounded worker(s)`，不能只看`resources.worker_threads`的全局池数。真实rustc/原生处理重叠、排队期间没有输出目录、实际nice及FIFO交接证据与r1同结构保存。

两次修复版本构建（初始候选及重编译负载）产生同一SHA `5b1a692677423372ccfa4a270dc0be4c10b9b68aa658e209060d6d3dd632b0d9`。资源探针数据与后续真实运行的线程数分别记录；不把探针中的1线程当作素材运行配置。

两次9张作为中等层冷/热范围的两份独立运行（独立alignment cache、未清OS页缓存），再对同一冻结候选执行快速层和其余三个中等层case：

```sh
python3 scripts/stack-lab/matrix.py --tier quick --binary task-4-capped-r2 --lane task-4 --label cap-verified-r2
python3 scripts/stack-lab/matrix.py --tier medium --case langyuan-10 --case wenyuan-10 --case lanting-framed --binary task-4-capped-r2 --lane task-4 --label cap-verified-r2
```

辅助快速层曾在等待build时被调度器主动延后，以保持演练现场清晰；取消记录保留在`matrix-cap-fix-r2-quick/`。随后上述正式快速层完整执行，不用取消记录充当通过结果。

## 验证结果

机器摘要：[task-4-two-slots.json](../benchmarks/task-4-two-slots.json)。原始报告、快照、构建日志、35项工具测试日志及最终复核均保存在`/Users/dp/Desktop/raw-editor/output/acceptance/task-4/`。

| 真实运行 | Rayon / 预处理 | 峰值RSS GiB | contended | TIFF与原预期 |
| --- | --- | --- | --- | --- |
| nine-cold | 5 / 5 | 7.862 | true | 完全一致 |
| nine-hot | 5 / 5 | 8.944 | true | 完全一致 |
| langyuan-10 | 5 / 5 | 10.702 | true | 完全一致 |
| wenyuan-10 | 5 / 5 | 10.584 | false | 完全一致 |
| lanting-framed | 5 / 5 | 11.876 | false | 完全一致 |

- r2全部调度检查通过：真实编译与原生处理重叠，nice≥10；第三个请求等待454.254秒，首个run释放后才取得槽，等待期间未创建输出目录。两次9张TIFF与cache分别落在独立目录和独立inode，原输入、参考、基线和用户已有设置的SHA保持。
- 两次9张TIFF均为`718a903dcd0bf102fd7ea7f2be652168ad972f18bc540f5854225ee2f1bb47e3`，QG pass/8ROI、闭环收敛、RAW18/fusion6、默认512MiB/1GiB缓存计数符合正式预期。
- 阆苑10、文苑10、兰亭45取景分别通过正式中等层断言，均首轮通过；兰亭严格及regression-only保护通过，纸面审计保持冻结值。未改正式预期或质量阈值。
- 5次实际运行最大RSS11.876GiB，均通过24GiB门。快速层7/7（含fmt/check、Rust库测试1095通过/0失败/44忽略及4项Node合同），35项Python工具测试通过。
- 原生池预算单测覆盖Rayon=1/2/5/8；冻结探针实际验证Rayon=1/预处理=1，真实5线程素材另按Stack_Report及原生日志核对。旧版本自动独占、手动包装旧版本进共享槽遭拒绝、错误探针拒绝冻结均有工具测试。
- L0已核对两次9张加其余三个case覆盖完整中等层5个范围，并独立重算摘要与保护；当前复看队列维持0/3，不新增用户画质确认或D4。

## 构建依赖与范围

Cargo以`--offline`解析依赖，但新worktree缺少忽略的ONNX Runtime时，现有build.rs仍会下载它并校验固定SHA。此次走了该路径；补齐后的文件与主工作区及源码固定的`2b885992d3d6fa4130d39ec84a80d7504ff52750027c547bb22c86165f19406a`一致，记录在`build-prerequisites.json`。不宣称整个构建没有网络访问。

本次修改了预处理线程预算，验证覆盖快速层、中等层5个范围及真实两槽演练；不外推睡觉45张、其他未验收全量素材、所有线程数或N8.2完整确定性矩阵。发生争用的耗时标contended，不用于性能结论。旧基线只作为像素比较基准保留，新版本通过后也不替换既有画质预期。

若任一9张TIFF不同于v28，保留失败、记录N8.2，不勾选完成，并按tasks.md要求交用户决定独占默认线程回退；不修改预期SHA或放宽门禁。
