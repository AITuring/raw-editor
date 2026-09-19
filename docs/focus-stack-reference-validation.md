# 连续画作参考验收记录（2026-09-18）

本文记录一次针对长幅平面画作的参考驱动验收，以及为该类输入保留在生产路径中的修复。
参考文件只用于建立可信几何和诊断对照；生产拼接器不会把参考图像像素复制进结果。

## 本次保留的算法修复

- 自动模式只有在投影模型的内点覆盖了足够大的画面区域时才接受单应性。重复的局部笔画即使
  能拟合出很小的残差，也不会再把整个平面拉成错误的投影。
- 多尺度特征不足时，先对固定大小的局部单元做有限对比度拉伸，再检测 FAST/BRIEF；描述子仍
  从原始分析图提取，因此局部拉伸只帮助发现低对比结构，不会改变匹配像素。
- 流式 mosaic 的色调场只从当前拍摄组已经拥有的重叠估计。拍摄组之间只允许低频、有限幅度
  的增益协调；它不能平均或覆盖源图高频细节。
- 所有权过渡按目标像素双线性采样分析掩膜，而不是给一个分析 cell 写入常量 alpha，避免
  方块边和宽接缝，同时保留被选中的源图细节。
- 最终画布保留全部真实源图覆盖的联合边界，不再用“最大无空洞矩形”裁掉画框。空白角落不以
  拉伸、镜像或生成纹理补齐。

实现位置：

- 自动模型选择和参考验收 harness：[`src-tauri/src/panorama_stitching.rs`](../src-tauri/src/panorama_stitching.rs)
- 流式所有权、色调场和 materialize：[`src-tauri/src/panorama_utils/mosaic.rs`](../src-tauri/src/panorama_utils/mosaic.rs)
- 低纹理局部对比特征：[`src-tauri/src/panorama_utils/processing.rs`](../src-tauri/src/panorama_utils/processing.rs)

## 参考驱动验收链路

测试构建中的 `reference_acceptance` 模块从已处理的 RAW 生成缩略图，然后由三个脚本完成：

1. `scripts/register-stack-reference.py` 用 SIFT/RANSAC 为每张源图估计到可信参考的单应性，并
   要求内点、空间支持、重投影误差和凸四边形检查全部通过。
2. `scripts/refine-stack-reference.py` 用局部对比度图和 DIS 双向光流估计稠密残差。有效区域
   要求前后向一致、位移受限，并按真实网格节点采样，避免通用 resize 带来的半 cell 偏移。
3. `scripts/estimate-reference-tone.py` 只拟合低频 RGB affine 系数
   `reference_low = gain * source_low + offset`，不写入参考像素。Rust harness 可选择逐源或按
   `group_id` 应用这些系数后，再走原有 RAW 加载、局部配准、清晰度评分和 ownership 渲染。

这些脚本需要 Python、NumPy 和 OpenCV；参考清单、生成的 JSON 和大尺寸源图属于本机验收材料，
不应提交到仓库。Rust 测试入口默认被 `#[ignore]` 标记，只在显式提供环境变量时运行。

## 阆苑女仙回归状态（未通过）

冻结输入为本机验收目录中连续的 84 张 `DSC_3680.NEF`–`DSC_3763.NEF`。几何参考为用户提供的
`未标题-9-yuanshi.jpg`；它的作用是注册和诊断对照，不是生产拼接器的像素来源。

此前参考驱动结果存在错位、分区接缝、明显失焦，并且裁掉了画框，不能作为验收通过或产品输出。
参考图只可作为离线诊断真值，不能参与生产配准、色调或像素合成。

该目录现在是通用算法的回归样例，而不是一次性制图任务。通过条件必须同时满足：自动模式仅输入
原始照片即可完成；完整画框保留；不依赖透视拉正；同一拍摄位置从原始分辨率选择最清晰焦平面；
不同拍摄位置的几何由多焦平面一致证据连接；全图和局部实像检查均无重影、虚焦块和可见接缝。
任何参考匹配合成图都不算通过。

## 验证命令

本次代码变更通过：

```text
cargo fmt --manifest-path src-tauri/Cargo.toml -- --check
cargo test --manifest-path src-tauri/Cargo.toml --lib panorama_utils::processing -- --nocapture
# 5 passed
cargo test --manifest-path src-tauri/Cargo.toml --lib panorama_utils::mosaic -- --nocapture
# 24 passed, 1 ignored
git diff --check
```

参考渲染的长时间命令需要用户本地 RAW 和参考文件，不作为常规 CI 测试；其输出目录也已加入
`.gitignore`。
