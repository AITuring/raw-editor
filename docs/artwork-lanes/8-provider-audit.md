# 任务8：注册 provider 离线对照

- 更新：2026-11-18；状态：**部分完成，依赖阻断**；分支`lane/task-8/provider-audit`，起点v1 `b1ce1b68`。
- 目标范围：固定阆苑0–2代理/原生平面，对照FAST/BRIEF、SIFT/AKAZE、ECC、LightGlue+DISK/ALIKED、RoMa；provider只提交候选，Rust几何门禁保持最终权威。
- 本次输出：[能力与fallback报告](../benchmarks/task-8-provider-audit.json)；[provider_audit.py](../../scripts/stack-lab/provider_audit.py)。

## 已完成

- 发现并核对现有Rust provider边界：schema=1，默认`rust_fast_brief`，可选Python provider失败时回退，fallback code包含`python_unavailable`；`ProviderThreadBudget`要求Rayon/OpenCV预算不超总线程且各自非零。
- Rust定向测试：6/6通过，包含optional provider fallback、payload schema/descriptor/坐标校验、稳定排序、线程预算。
- Python能力审计单元测试：2/2通过；报告明确写入`no fabricated scores`，不会把不可用 provider当成零分或通过。
- 当前环境能力：Rust FAST/BRIEF可用；`cv2`不可用，因此SIFT/AKAZE/ECC不可运行；`onnxruntime`不可用，因此LightGlue/DISK/ALIKED不可运行；没有RoMa sidecar或权重。
- 复用保存的阆苑0–2原生Rust路径证据：0→2为64 inliers、error 1.2600018px、hull support 0.3613；不把它标为provider收益。

## 结论与阻断

本次没有产生任何SIFT/AKAZE/ECC/learned-provider质量分数，因此不能完成PF.10的provider性能比较，也不修改生产默认或质量门槛。缺少可审计的provider包是唯一阻断条件；安装后需使用相同代理、输入摘要、预算和Rust几何门禁重跑0–2，记录召回、inliers、重投影P95、耗时、RSS、候选连通分量及fallback。

任务8保持未勾选；当前交付是安全边界和阻断证据，下一步先补齐固定provider运行环境，再继续PF.10。

## provider bundle r1：OpenCV 候选对照

在 `/private/tmp/raw-editor-provider-audit-venv` 建立了隔离 Python 3.13.3 venv，并固定安装：NumPy 2.2.6、opencv-python-headless 4.12.0.88、ONNX Runtime 1.20.1。wheel SHA、解释器 SHA、OpenCV build-info SHA 记录在 [task-8-provider-bundle-r1.json](../benchmarks/task-8-provider-bundle-r1.json)。SIFT、AKAZE、ECC API 均可导入；这不改变 Rust 默认 provider，也没有把临时 venv 写入仓库。

使用同一份阆苑 0–2 保存 plane，在长边 2048、OpenCV threads=1、固定 RNG seed 下完成候选对照：[langyuan-provider-compare-r1.json](../benchmarks/langyuan-provider-compare-r1.json)。结果摘要：

- 0→1：SIFT `707` inliers / P95 `1.512px`，AKAZE `12` / `1.279px`，ECC correlation `0.637`；Rust 保存证据为 `143` inliers / error `1.116px`。
- 0→2：SIFT `61` / P95 `1.892px`，AKAZE `5` / `1.370px`，ECC `0.144`；Rust 为 `64` / `1.260px`。
- 1→2：SIFT `1364` / P95 `1.626px`，AKAZE `36` / `1.100px`，ECC `0.616`；Rust 为 `583` / `0.998px`。

这些是 provider candidate metrics，不是 Stack_Report、QG 或 D2/D3 结果；候选没有通过 Rust sidecar 边界重新提交，也没有写入 pose、阈值或生产默认。ONNX Runtime 运行时已安装，但 LightGlue/DISK/ALIKED 权重和 sidecar 仍缺失；RoMa 仍不可用。因此任务8仍保持未勾选，下一步需要固定 learned-provider 权重、许可证和 sidecar schema 后，才能把完整 PF.10 对照接入 Rust 几何门。
