# 色彩管线契约

状态：2026-08-06 完成首轮代码审计与普通图片输入/核心预览 ICC 闭环。本文记录当前实现的
事实边界，不代表已经通过真实相机样片画质验收。

## 固定术语

- **sRGB 编码值**：使用 IEC 61966-2-1 分段传递函数、供显示或文件编码使用的数值。
- **线性 sRGB**：使用 sRGB/D65 原色，但数值与光强保持线性关系的工作数据。
- **输出 sRGB**：已经过 tone mapping、sRGB OETF 和最终显示域操作，可写入 8/16 位文件的数值。

CPU 的唯一标量实现是
[`color_management.rs`](../src-tauri/src/color_management.rs) 中的
`srgb_to_linear_channel` 与 `linear_to_srgb_channel`。WGSL 必须使用相同的 0.04045、
0.0031308、12.92、0.055 和 2.4 契约。`npm run color-contract:check` 会阻止 CPU 端出现第二套
实现，并检查 RAW 分支、主 shader、flare shader、GPU 输出和导出 ICC 边界。

普通图片的嵌入 RGB ICC 由 `image` 解码器读取，再由纯 Rust `moxcms` 转换到编码 sRGB；profile
解析上限为 4 MiB。矩阵型 profile 优先原地转换，LUT 型 profile 才分配临时输出缓冲。无 profile、
畸形 profile 或当前不支持的非 RGB profile 会记录警告并退回兼容的 sRGB 假设，不得影响文件解码。

## 当前数据流

| 阶段                     | 输入与处理                                                                                       | 输出契约                                                     | 当前验证                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------ |
| 马赛克 RAW 开发          | rawler 执行黑/白电平、白平衡、去马赛克与 `Calibrate`，RAW Editor 明确移除 `ProcessingStep::SRgb` | `RGBA32F` 线性 sRGB/D65；完整开发允许保留高于 1.0 的高光余量 | 静态边界检查；真实相机颜色仍待样片验收                 |
| LinearRaw                | 默认 `auto` 认为解码值已经线性；仅 `gamma`/`gamma_skip_calib` 模式先按标准 sRGB EOTF 解码        | 与普通 RAW 相同的线性浮点数据                                | 0.5 → 0.21404114 单元测试；已移除旧的 3.0 指数         |
| RAW 内嵌预览回退         | 内嵌 JPEG 被假定为 sRGB 编码，使用统一 EOTF 转线性；现有兼容路径随后乘以 0.4                     | `RGB32F` 线性 sRGB 近似值                                    | CPU 传递函数回归；0.4 只是兼容启发式，不是相机颜色校准 |
| JPEG/PNG/TIFF 等普通输入 | `image` 读取嵌入 RGB ICC，`moxcms` 转换为编码 sRGB；无效/缺失 profile 兼容回退为 sRGB 假设       | `RGB32F` 编码 sRGB，钳制到 0…1，尚未线性化                   | 内存生成的 Display P3 JPEG/PNG/TIFF 与回退测试         |
| 几何与修复区块           | 几何操作保持现有编码语义；修复区块按 `isSrgbEncoded` 决定是否通过统一 LUT 转线性                 | 交给 GPU 前，RAW/线性区块保持线性，普通图片保持 sRGB 编码    | `color-contract:check` 检查 LUT 来源                   |
| GPU 上传与入口           | CPU 图像上传到 `RGBA16Float`；普通图片执行 WGSL EOTF，RAW 直接通过                               | 线性 sRGB 浮点纹理                                           | RAW/非 RAW shader 分支静态回归                         |
| 线性编辑                 | 曝光、基础影调、局部调整、降噪、锐化等在线性数据上执行                                           | 线性 sRGB，允许 tone mapping 前保留高光余量                  | CPU/WGSL 公式与路由回归；节点级参考图仍需扩充          |
| 显示域编辑与输出         | AgX 或基础 RAW 映射产生显示值；随后应用 sRGB OETF、曲线、LUT、颗粒、裁切警告和抖动               | `RGBA8Unorm` 输出 sRGB，最终写入时钳制到 0…1                 | 主 shader 输出格式与钳制边界检查                       |
| WebView 预览             | 与导出相同的 GPU 输出像素编码为 JPEG，通过二进制 IPC 展示；核心预览、预设、LUT 与缩略图写入 ICC  | 带固定 sRGB v4 ICC 的 sRGB 编码字节                          | ICC 往返单元测试、二进制传输与静态边界检查             |
| 文件导出                 | 读取同一 GPU 输出，并按目标格式编码                                                              | JPEG、PNG、TIFF、WebP、JPEG XL 嵌入固定 sRGB v4 ICC          | 各格式尺寸与 ICC 往返单元测试                          |
| 原生显示                 | 显示 shader 原样采样输出纹理，WGPU surface 显式声明 `SurfaceColorSpace::Srgb`，由系统合成器映射  | 平台管理的 SDR sRGB surface                                  | 配置边界检查；不重复执行应用侧显示 profile 变换        |

快速预览、半分辨率编辑、100% ROI 和全分辨率导出改变的是分辨率、区域与调度，不得改变上表的
颜色空间语义。预览和导出都必须消费主 shader 的同一输出合同。

## 已知缺口

1. **嵌入输入 ICC**：当前闭环覆盖解码器能暴露的 RGB profile，限制为 4 MiB。CMYK/Gray 输入、
   profile 无法解析或超限时仍按 sRGB 兼容回退；CICP-only 输入尚未单独纳入合同。
2. **显示器 ICC**：日常 SDR 显示依赖 WebView/系统合成器根据 sRGB ICC 或 sRGB surface 做逐显示器
   映射。应用尚未提供自定义显示 profile、软打样、纸张/油墨模拟或显示 profile 调试覆盖。
3. 核心编辑 JPEG 已嵌入 ICC；旧 HDR/全景等非 basic 主链路的 PNG 预览仍依赖浏览器默认 sRGB，
   后续在这些模块进入 1.0 范围时统一迁移。
4. `apply_cpu_default_raw_processing` 只服务原图对比等旧 CPU 路径，是显示近似，不是主 RAW
   管线的参考实现。
5. 按当前开发决定，合法 Sony α7R V 真实样片的画质、60MP 性能和预览/导出人工一致性验收继续
   延后；本文和合成测试不替代该结论。

## 修改规则

- 新增 CPU 转换必须复用 `color_management.rs`，不得复制传递函数常量。
- 修改工作原色、传递函数、tone mapper 顺序或 RAW/非 RAW 标志含义时，必须同步更新本文、
  Rust 数值测试、WGSL 边界检查和预览/导出对照；若旧 sidecar 的像素语义发生变化，还必须增加
  schema 版本或迁移规则。
- 修改输入或显示 ICC 时，应保留非 sRGB profile 的确定性内存测试，并分别验证 WebView、原生
  WGPU surface 和各导出格式；不能只检查 ICC 标签存在。
