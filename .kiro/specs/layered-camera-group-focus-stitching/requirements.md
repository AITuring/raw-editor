# Requirements Document

## Introduction

本特性在 RAW Editor 现有拼图/景深合成链路（`panorama_stitching.rs`、`panorama_utils/stitching.rs`、
`panorama_utils/mosaic.rs`、`panorama_utils/seam_cut.rs`、`panorama_utils/photometric.rs`）之上，
把当前混在一条 mosaic 流程里的"机位内景深选择"和"机位间几何拼接"拆成显式的两层架构：

1. **机位层**：把视角一致、仅对焦点不同的多张 RAW 归为一个机位组，在原生分辨率完成组内配准与
   清晰度 ownership 合成，产出保留来源信息的 Virtual_Tile。
2. **拼接层**：只用 Virtual_Tile 参与机位间单应性配准、闭环联合优化、组级接缝和低频色调协调，
   最终输出一张处处清晰的大图。

这样机位间几何不再被某一张失焦原图牵引，机位内已确认的清晰度也不再被全局拼接过程破坏。

当前代码状态与本特性的差距：

- `FocusVirtualTile` 只携带 `image`、`tile_to_world`、`group_index`、`source_ids`，没有 ownership 掩膜、
  清晰度置信度、覆盖掩膜和 RAW 溯源；
- 虚拟瓦片路径本身已是默认路径，但瓦片级合成器的选择仍由环境变量决定：两者都未设置时走
  `stitching::progressive_seam_stitcher`，设置 `RAW_EDITOR_USE_STREAMING_VIRTUAL_TILE_MOSAIC` 走
  `mosaic::detail_preserving_mosaic`，设置 `RAW_EDITOR_USE_OWNERSHIP_VIRTUAL_TILE_STITCHER` 走
  `stitching::focus_tile_ownership_stitcher`，上述任一失败回退 `stitching::focus_stack_stitcher`；
  即默认合成器 `progressive_seam_stitcher` 不保证 ownership 硬归属，带 ownership 语义的
  `focus_tile_ownership_stitcher` 仍在环境变量之后。本特性需把 ownership 语义的合成器定为默认，
  把 `detail_preserving_mosaic` 与 `progressive_seam_stitcher` 降级为诊断开关，并保留
  `focus_stack_stitcher` 作为旧单层对照路径；
- `IMAGE_STACK_MAX_SOURCES` 在前端 `src/utils/imageStackPipeline.ts` 与后端
  `src-tauri/src/image_stack.rs` 均已是 500，但 `tests/image-stack-preview-contract.mjs` 仍断言 200、
  `README.md` 仍写 2–200，属于遗漏未更新，需一并统一到 500；
- 组级色调协调（`RAW_EDITOR_ENABLE_GLOBAL_PHOTOMETRIC`、`RAW_EDITOR_ENABLE_SPATIAL_TILE_EXPOSURE_GAIN`）
  同样是环境变量开关；
- 没有可自动判定"画质不劣于原图"的量化门槛，`tests/focus-stack-quality-contract.mjs` 只检查源码级常量与函数存在性；
- `docs/focus-stack-reference-validation.md` 记录的 `阆苑女仙` 84 张 NEF 回归仍为未通过。

冻结验收素材为 `/Users/dp/Downloads/书画拼图/阆苑女仙` 中连续的 84 张 `DSC_3680.NEF`–`DSC_3763.NEF`，
拍摄方式为从上到下、从右到左的二维扫描，组内包含同机位多焦点包围。该组素材在 Photoshop 中可以对齐，
因此几何失败不能归因于素材本身缺少重叠证据。参考图 `未标题-9-yuanshi.jpg` 只允许用于离线诊断对照，
不得进入生产配准、色调估计或像素合成。

本特性的硬约束：**最终输出画质不得劣于原图**。分辨率、锐度、噪声和色彩都必须通过可测量阈值验证，
而不是依赖人工观察结论。

## Glossary

- **Stack_Pipeline**：图像堆栈合成流程的顶层协调组件，负责调度分组、机位内合成、拼接层求解、
  输出与失败上报。
- **Source_RAW**：用户选择的原始 RAW 或位图输入文件，本特性验收素材为 Nikon NEF。
- **Capture_Station**（机位）：一组视角一致、仅对焦平面不同的 Source_RAW 集合；同一 Capture_Station
  内不存在有意的机位移动。
- **Station_Grouper**：从图像证据判定 Capture_Station 成员集合的组件。
- **Intra_Station_Registrar**：在单个 Capture_Station 内执行原生分辨率局部配准的组件。
- **Focus_Fuser**：在单个 Capture_Station 内用清晰度评分、像素一致性和图割决定每个区域由哪张
  Source_RAW 负责的组件。
- **Sharpness_Score**：在原生分辨率、经二项式低通后测量的相隔 4 个原生像素的梯度能量，包含亮度以及
  红绿、蓝绿颜色差通道，取值范围归一化到 `[0, 1]`。
- **Ownership_Map**：把 Virtual_Tile 的每个有效像素映射到唯一一个 Source_RAW 标识的整数掩膜。
- **Sharpness_Confidence**：Virtual_Tile 每个 ownership 单元的胜出 Sharpness_Score 与该单元次优候选
  Sharpness_Score 之差，按该单元共同梯度能量尺度归一化，取值范围 `[0, 1]`。
- **Textured_Pixel**：最终输出中所属 ownership 单元至少有 1 个候选的 Sharpness_Score 不低于 0.10 的
  非透明像素。此处的 Sharpness_Score 与本 Glossary 同名条目定义一致，即在原生分辨率、经二项式低通后
  测量的相隔 4 个原生像素的梯度能量，归一化到 `[0, 1]`；固定下界 0.10 表示该单元至少有一个候选达到
  归一化满量程 10% 的梯度能量，即存在可分辨的笔画或纸绢纹理边缘，而非均匀留白。低于该下界的像素
  记为平坦像素。
- **Coverage_Mask**：标记 Virtual_Tile 每个像素是否被至少一张 Source_RAW 的有效投影覆盖的二值掩膜。
- **Virtual_Tile**：一个 Capture_Station 的合成结果，由合成像素、Ownership_Map、Sharpness_Confidence、
  Coverage_Mask、`tile_to_world` 变换和 Source_RAW 溯源记录共同组成。
- **Virtual_Tile_Store**：创建、持久化和读取 Virtual_Tile 及其元数据的组件。
- **Capture_Topology_Model**：把 Capture_Station 建模为二维蛇形扫描网格并产出候选邻接关系的组件。
- **Candidate_Adjacency**：Capture_Topology_Model 输出的待验证机位对，只表示"值得尝试匹配"，
  不表示几何关系已成立。
- **Station_Relation**：通过匹配点、空间支持、尺度一致性和重叠复核后被接受的两个 Capture_Station
  之间的几何关系。
- **Station_Pose_Solver**：只使用 Virtual_Tile 或多焦点共识特征建立 Station_Relation 并求解机位位姿的组件。
- **Consensus_Feature**：在同一 Capture_Station 的至少 2 个焦平面上都被检出且位置一致的特征点。
- **Closure_Optimizer**：以最大支持生成树为初值、让所有可靠的水平、垂直和跨列闭环共同参与联合
  优化的组件。
- **Closure_Residual**：一个闭环约束在当前位姿解下的重投影误差，以世界坐标原生像素为单位。
- **Residual_Warp_Model**：在全局单应性之上描述局部残差形变的受约束模型，形式为分块网格或 APAP 类
  加权局部单应性。
- **Tone_Harmonizer**：只校正 Virtual_Tile 之间低频曝光和白平衡差异的组件。
- **Tile_Compositor**：在机位级执行接缝选择并把 Virtual_Tile 写入最终画布的组件。
- **Quality_Gate**：在输出前按可测量判据比较输出与 owner Source_RAW 的组件。
- **Owner_Region**：最终输出中由同一个 Source_RAW 负责的连通像素区域。
- **Local_Scale**：最终输出像素到其 owner Source_RAW 像素的雅可比行列式绝对值的平方根，
  数值 1.0 表示输出与源图像素密度相同。
- **MTF50_Normalized**：在倾斜边 ROI 上测得的 MTF50，按 Local_Scale 归一化到"每源图像素的周期数"。
- **Noise_Sigma**：在平坦 ROI 上经高通滤波后测得的像素标准差，按 owner Source_RAW 的同一 ROI 量纲计算。
- **Delta_E00**：CIEDE2000 色差。
- **Degradation_Manager**：在证据不足时选择降级路径或拒绝输出，并生成机器可读失败报告的组件。
- **Diagnostics_Recorder**：记录分组、位姿、闭环残差、ownership、色调场和 Quality_Gate 指标的组件。
- **Acceptance_Harness**：运行冻结素材回归并输出 JSON 指标报告的验收组件。
- **Stack_Report**：一次合成运行产出的机器可读 JSON 报告，包含分组、位姿、闭环、ownership 统计、
  Quality_Gate 指标和失败原因。
- **IMAGE_STACK_MAX_SOURCES**：图像堆栈单次输入上限常量，前端定义于
  `src/utils/imageStackPipeline.ts`，后端定义于 `src-tauri/src/image_stack.rs`，两端取值必须一致。

## Requirements

### Requirement 1: 机位分组识别

**User Story:** 作为扫描书画的摄影师，我希望系统自动识别哪些照片属于同一机位的多焦点包围，
这样我不需要手动拆分 84 张扫描照片。

#### Acceptance Criteria

1. WHEN Stack_Pipeline 收到数量在 2 与 `IMAGE_STACK_MAX_SOURCES`（含两端）之间的 Source_RAW，
   THE Station_Grouper SHALL 仅依据以下全部图像证据输出 Capture_Station 成员集合：两张 Source_RAW
   之间验证过的重叠内点匹配不少于 30 对、重叠区域局部归一化像素相关不低于 0.6、内点空间支持不低于
   重叠区域面积的 20%、两张之间的尺度比落在 `[0.98, 1.02]` 范围内。
2. THE Station_Grouper SHALL 把文件名数字、导入顺序和文件系统排序仅用于生成候选顺序，以及仅在两个
   候选的图像证据判定分数之差不超过 0.001 时作为稳定 tie-break。
3. WHEN 同一组 Source_RAW 被重命名或以至少 5 种不同随机顺序导入，THE Station_Grouper SHALL 输出与
   原顺序完全相同的 Capture_Station 成员集合划分。
4. WHEN 相邻候选之间累计画面中心位移超过画面长边的 0.02 倍，THE Station_Grouper SHALL 在该位置拆分
   Capture_Station。
5. THE Station_Grouper SHALL 支持单个 Capture_Station 包含 1 至 48 张 Source_RAW。
6. IF 某张 Source_RAW 与所有其他 Source_RAW 都不满足验收准则 1 的重叠证据判据，THEN THE
   Station_Grouper SHALL 把该 Source_RAW 标记为孤立、不并入任何 Capture_Station、继续完成其余
   Source_RAW 的分组，并在 Stack_Report 中记录该文件路径和孤立原因标识。
7. THE Station_Grouper SHALL 在 Stack_Report 中记录每个 Capture_Station 的成员文件路径、判定依据类型、
   判定分数、匹配内点数量和内点空间支持比例。
8. IF Source_RAW 数量少于 2 或超过 `IMAGE_STACK_MAX_SOURCES`，THEN THE Stack_Pipeline SHALL 拒绝启动
   分组、返回指明实际输入数量与允许范围的错误提示，并保持全部 Source_RAW 文件字节不变。
9. IF 某张 Source_RAW 无法解码，THEN THE Station_Grouper SHALL 把该 Source_RAW 从全部 Capture_Station
   中排除、继续完成其余 Source_RAW 的分组，并在 Stack_Report 中记录该文件路径和解码失败原因标识。
10. IF 某个候选 Capture_Station 的成员数量超过 48，THEN THE Station_Grouper SHALL 在累计画面中心位移
    最大的相邻候选处反复拆分该 Capture_Station，直到每个 Capture_Station 成员数量不超过 48，
    并在 Stack_Report 中记录每个拆分位置。

### Requirement 2: 机位内原生分辨率配准

**User Story:** 作为摄影师，我希望同机位不同焦点的照片在原生分辨率上精确对齐，
这样笔画不会因为对焦呼吸造成的细微缩放而产生双影。

#### Acceptance Criteria

1. WHEN 一个 Capture_Station 包含 2 张或更多 Source_RAW，THE Intra_Station_Registrar SHALL 选择该
   Capture_Station 内有效像素 Sharpness_Score 中位数最高的 Source_RAW 作为锚点帧（若最高值之差小于
   0.01，则按绝对路径升序取第一张），把锚点帧变换置为恒等，并把其余帧配准到锚点帧的原生像素坐标系。
2. WHEN 一个 Capture_Station 的锚点帧确定，THE Intra_Station_Registrar SHALL 先在实际分析图长边
   不超过 2048 个原生像素的分辨率完成 2 轮全局与局部配准，再在原生分辨率执行 patch refinement，
   并在 Stack_Report 中把 `analysis_long_side` 记录为本次路径实际使用的分析长边配置值（当前默认路径为
   1400），而不是记录 2048 这一上限。
3. THE Intra_Station_Registrar SHALL 以原生像素间距 `s` 采样 patch refinement 控制点，并使
   `s ≤ min(112, 8c)`，其中 `c` 是同一 Capture_Station 的 Focus_Fuser ownership 单元边长（Requirement 3
   第 1 条规定为 8 至 64 个原生像素）；每个控制点使用边长不小于 32 个原生像素的匹配窗口，使搜索可达
   范围不超过 64 个原生像素，并把实测 `s` 以原生像素为单位写入 Stack_Report。

   说明（修订依据与数学边界）：生产实现的分析坐标步长为 `NATIVE_REFINE_STEP = 16`，实际分析长边配置
   为 1400，所以原生间距为 `s = 16 × max(L / 1400, 1)`，其中 `L` 是机位平面原生长边。实测相机尺寸
   `L = 7088、8256、9504` 时，`s ≈ 81.0、94.4、108.6` 原生像素；对应生产 ownership 规则
   `c = clamp(floor(L / 512), 8, 64)` 得 `c = 13、16、18`，所以 `s/c ≈ 6.23、5.90、6.03`。
   因此 112 原生像素给出有意义的有限绝对上限，`8c` 给出相对 ownership 网格的有限稀疏度上限并为取整
   留出余量；这并不声称 109px 控制点间距与 18px ownership 单元具有相同密度。控制点数量与间距平方
   成反比：在 9504px 实测上把 108.6px 强制压到 18px，会把二维探针数放大
   `(108.6 / 18)² ≈ 36.4` 倍；本约束接受已测得的 81–109px，同时禁止间距随输入任意增大。窗口边长
   不小于 32 个原生像素与搜索可达范围不超过 64 个原生像素两条判据保持不变；当前实现的最细匹配窗口
   为 51 原生像素，粗遍搜索可达范围为 64 原生像素。

4. THE Intra_Station_Registrar SHALL 对每个局部匹配执行双向一致性检查，并拒绝正向与反向位移之和的
   模长超过 1.0 个原生像素的局部匹配。
5. THE Intra_Station_Registrar SHALL 对每个局部匹配执行邻域一致性检查，并拒绝与其 8 邻域已接受控制点
   位移中位数相差超过 4.0 个原生像素的局部匹配。
6. IF 某个局部匹配的对称重投影误差超过锚点帧原生像素长边的 0.01 倍，THEN THE Intra_Station_Registrar
   SHALL 拒绝该局部匹配，在该控制点位置保留全局模型结果，并把该控制点计入 Stack_Report 的被拒绝
   控制点数量。
7. IF 某张非锚点 Source_RAW 在原生分辨率 patch refinement 后的中位对称重投影误差不低于其全局模型
   结果的中位对称重投影误差，THEN THE Intra_Station_Registrar SHALL 对该帧整体回退为全局模型结果，
   并在 Stack_Report 中把该帧的局部配准状态记录为已回退。
8. IF 一个 Capture_Station 内某张 Source_RAW 的配准内点空间覆盖率低于 20%，THEN THE
   Intra_Station_Registrar SHALL 把该 Source_RAW 标记为配准失败，并在 Stack_Report 中记录该文件
   绝对路径与实测覆盖率，其中内点空间覆盖率定义为含已接受局部匹配的控制点单元面积之和除以锚点帧
   有效像素面积。
9. THE Intra_Station_Registrar SHALL 在 Stack_Report 中记录锚点帧文件绝对路径，以及每帧的变换矩阵、
   内点数量、内点空间覆盖率、被拒绝控制点比例、中位对称重投影误差和配准状态标识。

### Requirement 3: 机位内景深合成与 ownership

**User Story:** 作为摄影师，我希望同机位的多张照片合成出一张处处清晰的机位图，
并且每个区域的像素来自一张真实照片，而不是两张错位照片的平均。

#### Acceptance Criteria

1. THE Focus_Fuser SHALL 把 Capture_Station 的合成平面划分为正方形 ownership 单元，单元边长取
   8 至 64 个原生像素之间的值，且使合成平面长边不少于 512 个单元。
2. WHEN 一个 Capture_Station 的全部配准成功帧完成到锚点帧坐标系的重采样，THE Focus_Fuser SHALL
   在原生分辨率为每个 ownership 单元的中心和四角共 5 个位置测量 Sharpness_Score，采样窗口为边长
   32 个原生像素的正方形，且候选与当前底图在同一单元使用相同的采样位置和相同的采样窗口尺寸。
3. THE Focus_Fuser SHALL 为每个 ownership 单元计算候选与当前底图的像素不一致性，取值为两者在该单元
   低通后像素差绝对值的中位数并归一化到 `[0, 1]`；IF 某候选的不一致性超过 0.2 且其 Sharpness_Score
   高于当前底图，THEN THE Focus_Fuser SHALL 对该候选施加不小于 1.0 的选择代价惩罚，惩罚与归一化
   Sharpness_Score 同量纲。
4. THE Focus_Fuser SHALL 用图割在 ownership 网格上最小化由单元数据项（`1 − Sharpness_Score` 加不一致性
   惩罚）与相邻单元 owner 不同时按跨界像素差递增的平滑项组成的总代价，并使每个已覆盖单元得到恰好
   一个 owner。
5. IF 图割求解在 120 秒内未收敛，THEN THE Focus_Fuser SHALL 回退为逐单元选取代价最低的候选作为
   owner，并在 Stack_Report 中把该 Capture_Station 的 ownership 求解状态记录为降级。
6. THE Focus_Fuser SHALL 直接复制被选中 Source_RAW 的像素作为该 ownership 单元的输出，不对来自不同
   owner 的任何细节频带做加权平均，也不在 owner 边界引入过渡混合带。
7. IF 某些像素在投影后落在所有候选的有效覆盖之外，THEN THE Focus_Fuser SHALL 在 Coverage_Mask 中把
   这些像素标记为未覆盖、把其 alpha 置为 0，并在 Ownership_Map 中写入表示无 owner 的保留标识。
8. IF 某个 ownership 单元的所有候选 Sharpness_Score 都低于该 Capture_Station 内 Sharpness_Score 分布的
   10 百分位，THEN THE Focus_Fuser SHALL 把该单元标记为低清晰度区域，并在 Stack_Report 中记录该单元
   的世界坐标位置和以世界坐标原生像素计的面积。
9. THE Focus_Fuser SHALL 为每个已覆盖 ownership 单元输出取值在 `[0, 1]` 的 Sharpness_Confidence，
   其值为胜出候选与次优候选 Sharpness_Score 之差按该单元共同梯度能量尺度归一化的结果；IF 该单元
   只有 1 个候选，THEN THE Focus_Fuser SHALL 把该单元的 Sharpness_Confidence 置为 0。
10. WHERE 一个 Capture_Station 只包含 1 张配准成功的 Source_RAW，THE Focus_Fuser SHALL 跳过图割求解，
    把该 Source_RAW 的有效投影像素直接作为输出，并把 Ownership_Map 中全部已覆盖像素指向该
    Source_RAW 标识。
11. THE Focus_Fuser SHALL 使 Coverage_Mask 中被标记为已覆盖的每个像素在 Ownership_Map 中都有唯一一个
    Source_RAW 标识。

### Requirement 4: Virtual Tile 与元数据溯源

**User Story:** 作为摄影师，我希望机位合成结果保留来源信息而不是被提前压成普通 JPEG/TIFF，
这样后续拼接可以继续追溯到原始 NEF。

#### Acceptance Criteria

1. WHEN 一个 Capture_Station 完成景深合成，THE Virtual_Tile_Store SHALL 产出一个 Virtual_Tile，
   该 Virtual_Tile 同时包含合成像素、Ownership_Map、Sharpness_Confidence、Coverage_Mask、
   `tile_to_world` 变换，以及每个参与合成的 Source_RAW 的绝对路径、对该文件全部字节计算的 SHA-256
   和 ownership 像素数；溯源记录条目数等于参与合成的 Source_RAW 数量，且各条目 ownership 像素数
   之和等于 Ownership_Map 中已赋 owner 的像素数。
2. THE Virtual_Tile_Store SHALL 以 RGB 每通道不少于 32 位浮点保存 Virtual_Tile 合成像素，并在
   Virtual_Tile 元数据中把该像素的色彩编码标识记录为线性 sRGB 或显示编码 sRGB 之一。
3. THE Virtual_Tile_Store SHALL 以无损方式保存 Virtual_Tile 合成像素、Ownership_Map 和 Coverage_Mask，
   使读回结果与写入时的每个通道值、每个 owner 标识和每个覆盖标记逐元素完全相同。
4. THE Virtual_Tile_Store SHALL 把 Virtual_Tile 及其元数据写入应用私有数据目录下的版本化缓存，
   在缓存条目中记录 Source_RAW 路径集合、Source_RAW SHA-256 集合和流程版本标识，并把该缓存总占用
   限制在不超过 64 GiB，超限时按最后访问时间从早到晚整条删除缓存条目。
5. WHEN 同一 Capture_Station 的 Source_RAW 路径集合、SHA-256 集合和流程版本标识与某个已有缓存条目的
   三项记录完全一致，THE Virtual_Tile_Store SHALL 读取该缓存条目且不解码任何 Source_RAW。
6. THE Virtual_Tile_Store SHALL 使 Ownership_Map、Sharpness_Confidence 和 Coverage_Mask 与
   Virtual_Tile 合成像素使用同一像素坐标系且宽高完全相同，其中每个像素的 Sharpness_Confidence
   取其所属 ownership 单元在 `[0, 1]` 内的值。
7. THE Virtual_Tile_Store SHALL 使 Virtual_Tile 合成像素的有效范围等于 Coverage_Mask 已覆盖像素的
   联合边界。
8. THE Stack_Pipeline SHALL 以只读方式访问 Source_RAW，并使每个 Source_RAW 在合成成功、失败与取消
   三种结束路径下的 SHA-256 与读取前一致。
9. IF 某个 Capture_Station 不存在对应缓存条目，或其 Source_RAW 路径集合、SHA-256 集合、流程版本标识
   中任一项与已有缓存条目不一致，THEN THE Virtual_Tile_Store SHALL 重新合成该 Capture_Station 的
   Virtual_Tile，并以先写临时条目再整体替换的方式写入缓存，使任何时刻都不存在可被读取的部分写入条目。
10. IF 读取缓存条目时出现字段缺失、记录尺寸与实际像素尺寸不符或 SHA-256 集合无法复核，THEN THE
    Virtual_Tile_Store SHALL 视该条目为未命中、删除该条目并重新合成 Virtual_Tile，且在 Stack_Report
    中记录该 Capture_Station 与缓存失效原因标识。
11. IF 缓存写入因存储空间不足或目录不可写而失败，THEN THE Virtual_Tile_Store SHALL 保留内存中的
    Virtual_Tile 供后续拼接使用、不中断本次合成，并在 Stack_Report 中把该 Capture_Station 标记为
    缓存降级。

### Requirement 5: 拍摄拓扑与候选邻接

**User Story:** 作为摄影师，我按从上到下、从右到左扫描拍摄，我希望系统用这个拓扑先验来生成候选邻接，
但不要把文件名顺序当成几何结果。

#### Acceptance Criteria

1. WHEN Station_Grouper 输出的 Capture_Station 数量不少于 2，THE Capture_Topology_Model SHALL 按从上
   到下、从右到左的二维蛇形扫描顺序为每个 Capture_Station 输出一个非负整数行索引和一个非负整数
   列索引，且同一组行列索引最多分配给一个 Capture_Station。
2. THE Capture_Topology_Model SHALL 仅依据 Station_Grouper 输出的机位间估计中心位移推断行列索引，
   把水平估计中心位移超过单个机位有效宽度 0.5 倍的两个 Capture_Station 判为不同列，并把文件名顺序
   仅用作初始候选序列以及估计位移完全相同时的稳定 tie-break。
3. THE Capture_Topology_Model SHALL 为每个 Capture_Station 输出同列相邻、同行相邻和跨列相邻三类
   Candidate_Adjacency，其中同列相邻不超过 2 个、同行相邻不超过 2 个、跨列相邻不超过 4 个。
4. THE Capture_Topology_Model SHALL 使 Candidate_Adjacency 仅影响匹配尝试顺序和匹配尝试数量，
   不输出也不修改任何位姿参数或变换矩阵。
5. IF Candidate_Adjacency 数量超过单次运行 8 倍 Capture_Station 数量的搜索预算，THEN THE
   Capture_Topology_Model SHALL 按候选分数（估计重叠面积占单个机位有效面积的比例）降序保留预算内
   候选，并在 Stack_Report 中记录被截断的候选数量和截断时的最低保留分数。
6. WHEN Capture_Station 数量不超过 64，THE Capture_Topology_Model SHALL 输出全部机位对作为
   Candidate_Adjacency，且此时搜索预算不小于机位对总数，不触发截断。
7. IF 某个 Capture_Station 的行列索引无法由估计中心位移唯一确定，THEN THE Capture_Topology_Model
   SHALL 把该 Capture_Station 与其余全部 Capture_Station 在预算内配对为 Candidate_Adjacency，
   并在 Stack_Report 中把该机位的拓扑状态记录为索引歧义。
8. WHEN 同一组 Source_RAW 被重命名或以随机顺序导入，THE Capture_Topology_Model SHALL 输出与原顺序
   相同的行列索引和相同的 Candidate_Adjacency 集合。
9. THE Capture_Topology_Model SHALL 在 Stack_Report 中记录每个 Capture_Station 的行索引与列索引，
   以及每个 Candidate_Adjacency 的机位对、邻接类型和候选分数。

### Requirement 6: 机位间位姿图

**User Story:** 作为摄影师，我希望机位之间的几何关系只由已经处处清晰的机位图决定，
这样某一张失焦原图不会把整体几何拉偏。

#### Acceptance Criteria

1. THE Station_Pose_Solver SHALL 仅使用 Virtual_Tile 合成像素、Virtual_Tile 上提取的特征或
   Consensus_Feature 作为建立 Station_Relation 的匹配证据，并且仅在 Coverage_Mask 标记为已覆盖的
   像素上提取该证据。
2. IF 某条机位间匹配证据直接来自单张 Source_RAW 且未被 Consensus_Feature 确认，THEN THE
   Station_Pose_Solver SHALL 丢弃该证据、不将其计入任何 Candidate_Adjacency 的内点，并在
   Stack_Report 中记录被丢弃证据的数量与所属 Capture_Station。
3. WHEN 一个 Candidate_Adjacency 通过 RANSAC 拟合，THE Station_Pose_Solver SHALL 在接受该
   Station_Relation 之前要求内点数量不少于 24、内点重投影误差中位数不超过 3.0 个世界坐标原生像素，
   且两个 Capture_Station 之间的尺度比落在 `[0.95, 1.05]` 范围内。
4. THE Station_Pose_Solver SHALL 把内点空间支持比例定义为内点外接凸包面积与重叠区域面积之比，
   并要求被接受的 Station_Relation 的该比例不低于 20%。
5. IF 一个 Candidate_Adjacency 的内点空间支持比例低于 20%，THEN THE Station_Pose_Solver SHALL 拒绝
   该 Candidate_Adjacency、保留其余 Candidate_Adjacency 的匹配尝试，并在 Stack_Report 中以稳定的
   机器可读标识符记录拒绝原因为空间支持不足。
6. THE Station_Pose_Solver SHALL 为每个 Capture_Station 求解一个 8 自由度平面单应性位姿，不把尺度、
   旋转或透视自由度强制固定为恒等值，并使同一 Virtual_Tile 内任意两个已覆盖像素的 Local_Scale
   比值不超过 1.10。
7. IF 机位图存在不连通分量，THEN THE Station_Pose_Solver SHALL 拒绝输出完整拼接结果，在 Stack_Report
   中列出每个分量的 Capture_Station 成员与成员数量，并以稳定的机器可读标识符记录几何状态为不连通。
8. WHEN 一个 Candidate_Adjacency 通过内点数量、重投影误差与尺度比复核，THE Station_Pose_Solver SHALL
   在重叠像素上复核低频亮度均值相对差不超过 20%、边缘强度比落在 `[0.7, 1.4]` 范围内、边缘方向中位
   差不超过 10 度，并在任一判据未通过时拒绝该 Candidate_Adjacency。
9. IF 一个 Candidate_Adjacency 的内点数量少于 24、内点重投影误差中位数超过 3.0 个世界坐标原生像素、
   尺度比落在 `[0.95, 1.05]` 之外，或拟合单应性使 Virtual_Tile 四角投影成非凸四边形，THEN THE
   Station_Pose_Solver SHALL 拒绝该 Candidate_Adjacency、不写入任何位姿参数，并在 Stack_Report 中以
   稳定的机器可读标识符记录对应拒绝原因。
10. WHEN 机位位姿求解结束，THE Station_Pose_Solver SHALL 在 Stack_Report 中记录每个被接受
    Station_Relation 的机位对、内点数量、内点空间支持比例、尺度比和内点重投影误差中位数，
    以及被拒绝 Candidate_Adjacency 的数量与各拒绝原因标识符的计数。

### Requirement 7: 鲁棒闭环联合优化

**User Story:** 作为摄影师，我希望长幅二维扫描的累计漂移被所有可靠闭环共同纠正，
而不是沿一条生成树路径累积到最后一个机位。

#### Acceptance Criteria

1. THE Closure_Optimizer SHALL 以被接受的 Station_Relation 内点数量为边权、按内点数量降序构建最大支持
   生成树，在内点数量相同时按 Capture_Station 行列索引升序稳定 tie-break，并把该生成树的位姿解仅作为
   联合优化的初值。
2. THE Closure_Optimizer SHALL 使参与联合优化的约束数量等于被接受的水平、垂直和跨列 Station_Relation
   总数，不在优化前剔除任何被接受的 Station_Relation。
3. THE Closure_Optimizer SHALL 使用 M-estimator 鲁棒权重，使权重随 Closure_Residual 单调不增，
   Closure_Residual 不超过 2.0 个世界坐标原生像素的 Station_Relation 权重不低于其初始权重的 0.9 倍，
   Closure_Residual 超过 6.0 个世界坐标原生像素的 Station_Relation 权重不高于其初始权重的 0.1 倍。
4. THE Closure_Optimizer SHALL 把单个 Capture_Station 相对生成树初值的位姿修正限制为其四个角点在世界
   坐标中的位移均不超过该机位重叠区域较短边长度的 0.10 倍且不超过 256 个世界坐标原生像素，对超过该
   上限的修正按上限截断，并在 Stack_Report 中记录实际最大角点位移和被截断的 Capture_Station 数量。
5. WHEN 鲁棒加权总残差在相邻两次迭代之间的相对下降小于 0.0001，THE Closure_Optimizer SHALL 判定联合
   优化收敛，并在 Stack_Report 中记录参与约束数量、迭代次数、Closure_Residual 中位数和 Closure_Residual
   第 95 百分位。
6. THE Closure_Optimizer SHALL 要求联合解的 Closure_Residual 中位数不超过 2.0 个世界坐标原生像素。
7. IF 联合解的 Closure_Residual 中位数超过 2.0 个世界坐标原生像素，或联合优化在 100 次迭代内未满足
   收敛判据，THEN THE Closure_Optimizer SHALL 保留生成树初值、不改写任何 Capture_Station 的位姿，
   并在 Stack_Report 中把闭环状态记录为不可靠且记录触发回退的原因标识。
8. WHEN 联合解的 Closure_Residual 中位数不超过 2.0 个世界坐标原生像素，THE Closure_Optimizer SHALL
   使任意两个通过 Station_Relation 直接相连的 Capture_Station 在重叠区域的重投影误差第 95 百分位
   不超过 3.0 个世界坐标原生像素，并在 Stack_Report 中记录所有直接相连机位对中该第 95 百分位的最大值。
9. IF 存在任一通过 Station_Relation 直接相连的 Capture_Station 对，其重叠区域重投影误差第 95 百分位
   超过 3.0 个世界坐标原生像素，THEN THE Closure_Optimizer SHALL 保留生成树初值，并在 Stack_Report 中
   把闭环状态记录为不可靠且列出该 Capture_Station 对的行列索引。
10. IF 被接受的 Station_Relation 数量不超过 Capture_Station 数量减 1，THEN THE Closure_Optimizer SHALL
    跳过联合优化、直接输出生成树位姿解，并在 Stack_Report 中把闭环状态记录为无可用闭环约束。

### Requirement 8: 受约束的局部形变模型

**User Story:** 作为摄影师，我希望纸面轻微起伏能被吸收，但不希望局部形变模型把平整画面拉成错误形状。

#### Acceptance Criteria

1. WHILE 全局单应性在全部重叠区域的重投影误差第 95 百分位不超过 3.0 个世界坐标原生像素，
   THE Residual_Warp_Model SHALL 保持恒等映射，使任一网格节点的位移量为 0 个世界坐标原生像素。
2. WHEN 某个重叠区域的全局单应性重投影误差第 95 百分位超过 3.0 个世界坐标原生像素，
   THE Residual_Warp_Model SHALL 仅在该重叠区域启用节点间距为 64 个世界坐标原生像素、
   且每个方向不少于 4 个节点的分块网格或 APAP 类加权局部单应性，并在 Stack_Report 中记录该区域的
   世界坐标范围与启用前的实测误差值。
3. THE Residual_Warp_Model SHALL 限制网格中任意两个相邻节点的位移差不超过 8 个世界坐标原生像素。
4. THE Residual_Warp_Model SHALL 限制任一网格节点的位移量不超过 32 个世界坐标原生像素。
5. THE Residual_Warp_Model SHALL 仅在正向与反向映射的往返误差不超过 1.0 个世界坐标原生像素、
   且位移同时满足第 3 条与第 4 条约束的网格节点写入位移。
6. IF 启用局部形变后某个网格单元的重投影误差第 95 百分位不低于该单元在全局单应性下的对应值，
   THEN THE Residual_Warp_Model SHALL 在该单元回退为全局单应性，并在 Stack_Report 中记录该单元的
   世界坐标范围。
7. THE Residual_Warp_Model SHALL 使 Virtual_Tile 从机位坐标到世界坐标的总重采样次数不超过 1 次。
8. IF 某个网格节点未通过往返误差校验或位移约束，THEN THE Residual_Warp_Model SHALL 用距离不超过
   3 个节点的有效邻域节点外推该节点位移；IF 该范围内不存在有效邻域节点，THEN THE
   Residual_Warp_Model SHALL 把该节点位移置为 0 个世界坐标原生像素。
9. THE Residual_Warp_Model SHALL 使局部形变位移在距启用区域边界 128 个世界坐标原生像素范围内单调
   衰减至 0，并使启用区域之外的像素保持全局单应性结果。
10. IF 某个需要启用局部形变的重叠区域内通过往返误差校验的匹配点少于 16 个，THEN THE
    Residual_Warp_Model SHALL 在该区域保持恒等映射，并在 Stack_Report 中记录该区域范围与原因为
    局部形变证据不足。

### Requirement 9: 组级低频色调校正

**User Story:** 作为摄影师，我希望机位之间的曝光和白平衡差异被平滑掉，但笔画的清晰度归属和高频细节
不受色调算法影响。

#### Acceptance Criteria

1. THE Tone_Harmonizer SHALL 仅使用同时满足以下全部条件的对应像素估计 Virtual_Tile 之间的增益与
   偏移：该像素对位于已被接受的 Station_Relation 的重叠区域内、在两个 Virtual_Tile 的 Coverage_Mask
   中均标记为已覆盖、且两侧归一化亮度均落在 `[0.02, 0.98]` 范围内；并要求每个 Virtual_Tile 对参与
   估计的有效样本数不少于 1024。
2. THE Tone_Harmonizer SHALL 仅在低频频带上应用校正，该低频频带为以标准差不小于 64 个世界坐标原生
   像素的低通滤波得到的亮度与 R、G、B 分量，并使输出的高频残差频带逐像素等于 owner Source_RAW 的
   对应高频残差。
3. THE Tone_Harmonizer SHALL 限制单个 Virtual_Tile 的亮度增益在 `[0.8, 1.25]` 范围内、每个 RGB 通道
   增益在 `[0.8, 1.25]` 范围内，且每个通道偏移的绝对值不超过归一化亮度满量程的 0.02。
4. THE Tone_Harmonizer SHALL 在估计增益前对每个 Virtual_Tile 的重叠样本执行 MAD 一致性检查，
   并排除与样本中位数偏差超过 3 倍 MAD 的样本。
5. THE Tone_Harmonizer SHALL 在 Ownership_Map 与 Tile_Compositor 的接缝选择全部完成后再应用色调
   校正，且不向接缝选择提供任何已应用色调校正的像素。
6. THE Tone_Harmonizer SHALL 使色调校正后的 Ownership_Map 与校正前的 Ownership_Map 逐像素一致。
7. WHERE 某区域的对应像素不满足第 1 条的有效样本条件，THE Tone_Harmonizer SHALL 对该区域使用恒等
   增益与零偏移，保留该区域 owner Source_RAW 的原始曝光。
8. THE Tone_Harmonizer SHALL 使任意两个相邻 Owner_Region 边界两侧各 16 个世界坐标原生像素带内的
   低频均值 Delta_E00 不超过 1.5。
9. IF 求解得到的增益或偏移超出第 3 条规定的范围，THEN THE Tone_Harmonizer SHALL 把该增益或偏移
   截断到最近的边界值，并在 Stack_Report 中记录该 Virtual_Tile 标识、求解值与截断值。
10. IF 某个 Virtual_Tile 对经 MAD 一致性检查后保留的有效样本数少于 1024，THEN THE Tone_Harmonizer
    SHALL 把该 Virtual_Tile 对视为无重叠证据、对该对使用恒等增益与零偏移，并在 Stack_Report 中
    记录该对标识与保留样本数。
11. IF 某相邻 Owner_Region 边界带内的低频均值 Delta_E00 超过 1.5，THEN THE Tone_Harmonizer SHALL 在
    Stack_Report 中记录该 Owner_Region 对标识、实测 Delta_E00 与该边界的世界坐标位置，并把色调
    协调状态标记为降级。

### Requirement 10: 机位级接缝与最终输出

**User Story:** 作为摄影师，我希望最终大图保留完整画框、每个像素可追溯，并且输出格式不引入额外画质损失。

#### Acceptance Criteria

1. THE Tile_Compositor SHALL 仅在两个 Virtual_Tile 的 Coverage_Mask 都标记为已覆盖的重叠像素上选择机位间接缝，接缝代价同时由候选像素到各自有效覆盖边界的距离与重叠像素不一致性决定，对距有效覆盖边界不足 16 个世界坐标原生像素的候选像素施加不小于 1.0 的代价惩罚，并取全局代价最低的接缝路径。
2. THE Tile_Compositor SHALL 使最终画布边界等于全部 Virtual_Tile 的 Coverage_Mask 已覆盖像素在世界坐标中的联合轴对齐外接边界，支持画布长边不少于 262,144 个世界坐标原生像素，且不做额外裁切或外扩。
3. THE Tile_Compositor SHALL 保留不被任何 Virtual_Tile 有效覆盖的像素为完全透明，不使用镜像、延拓或生成纹理补齐，并且不把任一 Virtual_Tile 中 Coverage_Mask 标记为未覆盖的像素写入最终画布。
4. THE Tile_Compositor SHALL 在与最终画布同尺寸、同坐标系的输出 Ownership_Map 中为每个非透明像素记录唯一一个 Source_RAW 标识，且该标识等于该像素所属 Virtual_Tile 的 Ownership_Map 在对应位置的标识。
5. THE Stack_Pipeline SHALL 以每通道不少于 16 位整数的显示编码 sRGB 精度输出最终结果，并在输出文件中嵌入 sRGB ICC。
6. WHERE 用户选择支持 alpha 的输出格式，THE Stack_Pipeline SHALL 把未覆盖像素的 alpha 写为完全透明、把已覆盖像素的 alpha 写为完全不透明，且不因写入 alpha 改变已覆盖像素的颜色通道值。
7. THE Stack_Pipeline SHALL 使最终结果与其预览携带同一个结果标识，并使预览仅由最终结果的规范化显示编码像素降采样得到，且预览与最终结果在任意对应 ROI 的低频均值 Delta_E00 不超过 1.0。
8. WHEN 一次合成运行结束，无论结果为成功输出、降级输出还是被 Quality_Gate 阻止导出，THE Stack_Pipeline SHALL 在 30 秒内写出一份 Stack_Report。
9. IF 某相邻 Virtual_Tile 对的有效重叠宽度小于 32 个世界坐标原生像素，THEN THE Tile_Compositor SHALL 沿该重叠区域有效覆盖的中线选择接缝，并在 Stack_Report 中记录该机位对标识与实测重叠宽度。
10. IF 用户选择的输出格式不支持每通道 16 位或不支持 alpha，THEN THE Stack_Pipeline SHALL 在写出最终结果前返回指明位深或透明度将被降级的提示，并在 Stack_Report 中记录实际输出位深与 alpha 保留状态。
11. IF 联合覆盖边界的长边超过支持的画布上限，THEN THE Tile_Compositor SHALL 拒绝写出最终结果，并在 Stack_Report 中记录实测画布尺寸与该上限。

### Requirement 11: 输出画质不劣于原图的可测量判据

**User Story:** 作为摄影师，我需要客观证明拼接结果在分辨率、锐度、噪声和色彩上都不比原图差，
而不是依赖肉眼判断。

#### Acceptance Criteria

1. WHEN Tile_Compositor 完成最终画布且导出尚未开始，THE Quality_Gate SHALL 按以下确定性规则选取
   ROI 集合：候选 ROI 为边长 512 个原生像素的正方形，且完全落在单个 Owner_Region 内部、内部非透明
   像素比例为 100%、四边与该 Owner_Region 边界的距离不小于 16 个原生像素；对每个面积不小于 4 倍
   ROI 面积的 Owner_Region 至少选取 1 个 ROI；ROI 总数不少于 32 个且不超过 256 个；候选数超过 256 时
   按 ROI 左上角世界坐标行优先顺序等间隔抽取。
2. THE Quality_Gate SHALL 把每个被选取 ROI 与其 owner Source_RAW 中的同一场景区域配对，配对时不对
   输出 ROI 做任何重采样，仅对 owner Source_RAW 参考区域执行不超过 1 次重采样以对齐到 ROI 像素网格，
   并要求配对后残余对齐误差不超过 0.5 个原生像素。
3. THE Quality_Gate SHALL 以每个被选取 ROI 内 Local_Scale 的中位数计量，要求该中位数不低于 0.98，
   且该 ROI 内 Local_Scale 不低于 0.95 的像素比例不小于 99%。
4. THE Quality_Gate SHALL 以最终输出的非透明像素计数作为总有效像素数，以全部 Source_RAW 投影到世界
   坐标后的唯一覆盖面积（重叠区域只计一次）作为基准，并要求该计数不低于基准的 0.98 倍。
5. THE Quality_Gate SHALL 把满足以下全部条件的 ROI 判定为倾斜边 ROI：含至少 1 条长度不小于 128 个
   原生像素的直线边、该边与最近像素轴夹角在 3 度至 15 度之间、边两侧低频亮度对比度不小于满量程的
   20%、边两侧各 32 个原生像素内不存在其他满足上述对比度条件的边。
6. THE Quality_Gate SHALL 在每个倾斜边 ROI 上测量 MTF50_Normalized，并要求输出 ROI 的
   MTF50_Normalized 不低于配对 owner Source_RAW 参考区域 MTF50_Normalized 的 0.93 倍。
7. THE Quality_Gate SHALL 以 Glossary 中 Sharpness_Score 所用的梯度度量在亮度通道上计算 ROI 平均梯度
   能量并按 Local_Scale 归一化，要求每个被选取 ROI 的该归一化值不低于配对 owner Source_RAW 参考区域
   同一归一化值的 0.95 倍。
8. THE Quality_Gate SHALL 把满足以下全部条件的 ROI 判定为平坦 ROI：ROI 内低频亮度标准差不超过满量程
   的 2%、不存在满足倾斜边条件的边、非透明像素比例为 100%。
9. THE Quality_Gate SHALL 在每个平坦 ROI 上测量 Noise_Sigma，并要求输出 Noise_Sigma 与配对 owner
   Source_RAW 参考区域 Noise_Sigma 的比值落在 `[0.85, 1.15]` 范围内。
10. THE Quality_Gate SHALL 在最终输出的 sRGB 色空间下计算每个被选取 ROI 输出低频均值与配对 owner
    Source_RAW 参考区域低频均值之间的 Delta_E00，并要求该 Delta_E00 不超过 2.0。
11. THE Quality_Gate SHALL 在相邻 Owner_Region 公共边界上每 256 个原生像素长度取 1 个测量点，在该点
    边界两侧各 16 个原生像素带内检测低频对比度不小于满量程 15% 的边缘，仅把在两侧都被检出且方向差
    不超过 10 度的边缘作为同一笔画边缘配对，并要求配对边缘的亚像素位置偏差第 95 百分位不超过 1.5 个
    原生像素、最大值不超过 3.0 个原生像素。
12. THE Quality_Gate SHALL 以最终输出的全部 Textured_Pixel 为统计范围，要求 Sharpness_Confidence
    低于 0.05 的像素占 Textured_Pixel 数量的比例不超过 1%，并把 Textured_Pixel 数量、被排除的平坦
    像素数量、Textured_Pixel 中 Sharpness_Confidence 低于 0.05 的像素占比和被排除平坦像素中
    Sharpness_Confidence 低于 0.05 的像素占比一并写入 Stack_Report。

    说明（修订依据，实测）：本条原文以全部非透明像素为统计范围，实测不可满足。按 Requirement 3
    第 9 条的定义，Sharpness_Confidence 为 `(s_best − s_second) / joint_gradient_scale`，纸绢底大片
    留白处两帧锐度几乎相同，该比值天然趋 0；三个机位实测 Sharpness_Confidence 低于 0.05 的非透明
    像素占比为 37.4%、34.9%、51.2%，与 1% 相差两个数量级。这些区域选哪一帧对输出像素无影响，低置信度
    在那里是无害的。本条的原意是"不要留下大片分不清哪帧更清晰的**有内容**区域"，因此把统计范围限定为
    Textured_Pixel。Stack_Report 同时记录有纹理与平坦两个子集的像素数和各自低置信度占比，使该判据
    可复核，且无法通过缩小统计范围规避。

13. IF 某个判据在某个 ROI 或某个边界测量点上不满足其测量前置条件，THEN THE Quality_Gate SHALL 把该
    测量项标记为不可测量，既不计入通过也不计入失败，并在 Stack_Report 中记录该测量项的判据名称、
    世界坐标位置、不可测量原因和原因类别。原因类别分两类：内容不适用（不属于倾斜边 ROI、不属于
    平坦 ROI、边界两侧无可配对边缘）与技术性不可测（owner Source_RAW 不可解码、配对残余对齐误差
    超过 0.5 个原生像素）。
14. THE Quality_Gate SHALL 按以下规则确定判据结论，结论为证据不足时阻止导出，并在 Stack_Report 中
    记录该判据的结论、可测量与不可测量计数及原因：
    (a) 对逐 ROI 或逐测量点的判据，应测量项总数不含内容不适用的测量项；技术性不可测的测量项占
        应测量项总数的比例超过 20% 时，结论为证据不足。
    (b) `local_scale_median`、`local_scale_pixel_ratio`、`gradient_energy_normalized`、
        `roi_delta_e00` 的可测量测量项少于 8 个时，结论为证据不足。
    (c) `mtf50_normalized`、`noise_sigma_ratio`、`boundary_stroke_alignment` 的可测量测量项少于
        8 个、(a) 不成立且没有未通过的可测量测量项时，结论为不适用，既不计为通过也不阻止导出；
        `mtf50_normalized` 不适用时锐度由 `gradient_energy_normalized` 判定。只要存在未通过的可测量
        测量项，该判据按第 15 条为未通过，不论可测量项数量多少。
    (d) `effective_pixel_count` 与 `sharpness_confidence_coverage` 是整幅输出的单项统计，不受可测量
        项数量下限约束；其统计基准为空（唯一覆盖面积为 0 或 Textured_Pixel 数量为 0）时结论为证据
        不足。两者在 Stack_Report 中另按 Capture_Station 记录分项数值，分项数值不影响结论。
15. IF 任一判据存在未通过的可测量测量项，THEN THE Quality_Gate SHALL 阻止写出最终输出文件、保留
    诊断预览与已生成中间产物、保持全部 Source_RAW 字节不变，向调用方返回指明未通过判据的错误提示，
    并在 Stack_Report 中记录未通过判据名称、实测值、阈值、对应 ROI 或测量点的世界坐标位置和 owner
    Source_RAW 路径。
16. THE Quality_Gate SHALL 把全部实测指标写入 Stack_Report，包含通过的判据、每个判据的阈值、可测量
    测量项数量、不可测量测量项数量和最终判定结论。
17. WHEN 同一组 Source_RAW 与同一组参数被重复运行，THE Quality_Gate SHALL 选取世界坐标与顺序完全
    相同的 ROI 集合，并输出相同的判定结论。

### Requirement 12: 失败与退化处理

**User Story:** 作为摄影师，我希望在证据不足时得到明确的失败原因和可操作提示，而不是一张看起来完整
但内部错误的大图。

#### Acceptance Criteria

1. IF 某个 Capture_Station 内所有非锚点帧都被 Intra_Station_Registrar 标记为配准失败，THEN THE
   Degradation_Manager SHALL 把该 Capture_Station 降级为仅使用有效面积内中位 Sharpness_Score 最高的
   单张 Source_RAW（中位 Sharpness_Score 完全相同时取绝对路径字典序最小者），使该 Virtual_Tile 的
   Ownership_Map 全部指向该 Source_RAW，并在 Stack_Report 中把该机位标记为单帧降级。
2. IF 某个 Capture_Station 内有 1 张或更多非锚点帧被标记为配准失败且配准成功的帧数不少于 2，
   THEN THE Degradation_Manager SHALL 仅用配准成功的帧完成该机位景深合成，并在 Stack_Report 中列出
   每个被排除 Source_RAW 的绝对路径与排除原因标识符。
3. IF 闭环状态被记录为不可靠，THEN THE Degradation_Manager SHALL 使用生成树位姿继续输出，在
   Stack_Report 中把几何置信度标记为降级，并保持 Requirement 11 的全部 Quality_Gate 判据仍然生效。
4. IF 机位图存在不连通分量，THEN THE Degradation_Manager SHALL 拒绝写出最终结果文件，写出记录每个
   分量 Capture_Station 成员绝对路径的 Stack_Report，并向用户返回错误提示，该提示指明不连通分量
   数量以及按连续场景重新分组与增加重叠两类可操作动作。
5. IF 某张 Source_RAW 相对候选 Capture_Station 锚点帧的单应性内点空间支持比例低于重叠区域面积的
   20%，或其内点中位对称重投影误差超过画面长边的 0.01 倍，THEN THE Degradation_Manager SHALL 拒绝
   把该 Source_RAW 并入任何 Capture_Station，并在 Stack_Report 中记录该文件绝对路径、实测内点空间
   支持比例、实测中位对称重投影误差和拒绝原因标识符。
6. IF Quality_Gate 阻止导出，THEN THE Degradation_Manager SHALL 拒绝写出最终结果文件，保留诊断预览
   和 Stack_Report，并向用户返回错误提示，该提示包含每个未通过判据名称及其对应 ROI 的世界坐标位置。
7. WHEN 任一失败或降级路径生效，THE Degradation_Manager SHALL 在 Stack_Report 中为该路径写入一个
   机器可读失败原因标识符，该标识符仅由 ASCII 小写字母、数字和下划线组成、长度不超过 64 个字符，
   且对同一失败原因在重复运行之间保持一致。
8. THE Degradation_Manager SHALL 使任一失败或降级路径结束后每个 Source_RAW 的绝对路径存在性与
   SHA-256 与本次运行开始前一致。
9. WHEN 同一次运行中有 2 条或更多降级或拒绝路径生效，THE Degradation_Manager SHALL 在 Stack_Report
   中记录全部生效路径的失败原因标识符，并使拒绝输出优先于降级输出决定本次运行结果。
10. IF Degradation_Manager 拒绝输出，THEN THE Degradation_Manager SHALL 使输出目录中不残留本次运行
    部分写入的最终结果文件，并保留已写出的 Stack_Report 与诊断预览。

### Requirement 13: 诊断与可追溯性

**User Story:** 作为开发者，我需要定位某个模糊区域由哪个机位、哪张 NEF 负责，才能持续优化算法。

#### Acceptance Criteria

1. WHERE 用户在设置中启用堆栈诊断，WHEN 一次合成运行结束（无论成功或失败），THE Diagnostics_Recorder
   SHALL 把每个 Capture_Station 的成员集合、每帧变换、局部残差场、Ownership_Map、Sharpness_Confidence、
   Coverage_Mask 和低频色调场写入用户指定目录，且每项输出都携带其所属 Capture_Station 标识。
2. THE Diagnostics_Recorder SHALL 使诊断输出使用与最终输出相同的世界坐标系和相同的像素原点，
   坐标与尺寸以世界坐标原生像素为单位，并记录最终有效裁切区域的左上角坐标、宽度和高度。
3. WHEN 用户指定一个长边不超过 4096 个世界坐标原生像素的世界坐标 ROI，THE Diagnostics_Recorder SHALL
   在 60 秒内导出该 ROI 内每个候选 Source_RAW 的重采样结果、选择掩膜、逐帧 Sharpness_Score 和最终
   ownership，且各项输出与该 ROI 使用相同尺寸和相同坐标原点。
4. THE Diagnostics_Recorder SHALL 仅把诊断输出写入用户指定目录，不在 Source_RAW 所在目录创建、修改或
   删除任何文件，并保持每个 Source_RAW 文件字节不变。
5. WHILE 堆栈诊断处于关闭状态，THE Diagnostics_Recorder SHALL 使完整尺寸诊断缓冲的分配数量为 0，
   且不写入任何诊断文件。
6. WHEN 用户指定最终输出中的一个非透明像素坐标，THE Diagnostics_Recorder SHALL 返回该像素所属的
   Capture_Station 标识、owner Source_RAW 绝对路径、该像素的 Sharpness_Confidence 取值和
   Coverage_Mask 取值。
7. IF 用户指定目录不存在、不可写或诊断写入过程中失败，THEN THE Diagnostics_Recorder SHALL 停止后续
   诊断写入、保留已产出的最终输出与 Stack_Report，并给出指明诊断写入失败及目标目录的错误提示。
8. IF 用户指定的世界坐标 ROI 长边超过 4096 个世界坐标原生像素、完全落在最终有效裁切区域之外，或与
   Coverage_Mask 已覆盖像素无交集，THEN THE Diagnostics_Recorder SHALL 拒绝该次导出、不写入任何部分
   结果，并给出指明 ROI 无效原因的错误提示。

### Requirement 14: 资源边界与确定性

**User Story:** 作为摄影师，我希望 84 张 45MP NEF 的合成在我的 32 GB Mac 上稳定完成，并且重复运行
得到相同结果。

#### Acceptance Criteria

1. WHEN Stack_Pipeline 以 84 张不少于 45 兆像素的 Source_RAW 为输入运行，THE Stack_Pipeline SHALL 使
   从接受输入到写出 Stack_Report 期间、以不低于 1 Hz 采样的进程常驻内存最大值不超过当前生效的内存
   门槛，并把实测峰值常驻内存、生效门槛和采样次数写入 Stack_Report。
2. THE Stack_Pipeline SHALL 把内存门槛默认值设为 24 GiB（基于 32 GB 物理内存机型的估计值），支持在
   4 GiB 至物理内存 0.75 倍之间配置，并支持按物理内存 0.75 倍自动校准，且把生效门槛来源记录为
   默认值、用户配置或自动校准三者之一。
3. IF 实测峰值常驻内存超过生效内存门槛，THEN THE Stack_Pipeline SHALL 中止该次运行、不写出任何部分
   最终结果文件、删除该次运行产生的临时文件，并在 Stack_Report 中以稳定的机器可读标识符记录内存
   超限原因、生效门槛和实测峰值。
4. THE Stack_Pipeline SHALL 使任意时刻同时常驻内存的完整尺寸 Virtual_Tile 数量不超过 2 个，
   并在 Stack_Report 中记录该次运行观测到的最大同时常驻 Virtual_Tile 数量。
5. THE Stack_Pipeline SHALL 使分块处理、落盘和缓存复用等降低内存占用的措施不改变最终输出的像素尺寸、
   位深、Ownership_Map 和输出文件字节内容。
6. WHEN 同一组 Source_RAW 与同一组参数在同一构建版本和同一平台上被重复运行，THE Stack_Pipeline SHALL
   输出逐字节一致的最终结果文件，且该一致性不受并行线程数量、Virtual_Tile 缓存命中与否和文件导入
   顺序影响。
7. WHILE 合成任务运行，THE Stack_Pipeline SHALL 以不超过 2 秒的间隔更新当前阶段名称、已完成机位数和
   总机位数，并在收到取消请求后 1 秒内确认该请求已被接受。
8. WHEN 用户取消合成任务，THE Stack_Pipeline SHALL 在 5 秒内停止解码新的 Source_RAW、删除该次运行
   产生的临时文件、不写出部分最终结果文件、保持 Source_RAW 文件字节与已有 Virtual_Tile 缓存条目不变，
   并在 Stack_Report 中以稳定的机器可读标识符记录取消原因。
9. THE Stack_Pipeline SHALL 在网络接口不可用的条件下完成全部合成功能，且不发起任何出站网络请求。

### Requirement 15: 默认路径与回归验收

**User Story:** 作为开发者，我需要分层机位合成成为默认路径，并由冻结素材回归防止退化。

#### Acceptance Criteria

1. WHEN Stack_Pipeline 开始一次合成运行且 Station_Grouper 输出的 Capture_Station 数量不少于 2，
   THE Stack_Pipeline SHALL 选择 Virtual_Tile 分层合成路径，并在 Stack_Report 中记录所选路径的
   稳定机器可读标识符。
2. THE Stack_Pipeline SHALL 在默认构建配置下使 Virtual_Tile 分层合成路径、组级色调协调和
   Quality_Gate 生效，且不把任何环境变量、命令行参数或配置项作为这三者的启用前提。
3. WHERE 开发者启用对照旧路径的诊断开关，THE Stack_Pipeline SHALL 使用旧的单层 mosaic 路径，
   并在 Stack_Report 中记录所选路径的稳定机器可读标识符。
4. WHEN Acceptance_Harness 以 `阆苑女仙` 目录中连续的 84 张 `DSC_3680.NEF`–`DSC_3763.NEF` 作为
   唯一输入运行，THE Acceptance_Harness SHALL 在不使用任何参考图像、不要求用户手动分组、不访问网络
   且峰值常驻内存不超过 Requirement 14 规定的生效内存门槛的条件下产出最终结果文件和 Stack_Report。
5. THE Acceptance_Harness SHALL 要求该次运行的 84 张 Source_RAW 全部被分配到某个 Capture_Station，
   被标记为孤立的 Source_RAW 数量为 0，且机位图为单一连通分量。
6. THE Acceptance_Harness SHALL 要求该次运行通过 Requirement 11 的全部 Quality_Gate 判据，
   未通过判据数量为 0。
7. THE Acceptance_Harness SHALL 要求该次运行最终输出的有效边界面积不低于全部 Source_RAW 按求解位姿
   投影后联合边界面积的 0.98 倍。
8. THE Acceptance_Harness SHALL 把 Stack_Report 以 JSON 文件写入调用方指定的临时目录，且不向
   Source_RAW 所在目录写入或修改任何文件。
9. THE Stack_Pipeline SHALL 使常规回归套件在不读取任何 Source_RAW、不访问网络的条件下验证分组稳定性、
   Virtual_Tile 元数据完整性、闭环约束参与度、Residual_Warp_Model 约束边界、色调增益范围和
   Quality_Gate 阈值常量。
10. THE Stack_Pipeline SHALL 使对照旧单层 mosaic 路径的诊断开关在默认构建中处于关闭状态。
11. IF Station_Grouper 输出的 Capture_Station 数量少于 2，THEN THE Stack_Pipeline SHALL 仅执行机位层
    合成并输出该 Capture_Station 的 Virtual_Tile 结果，不执行机位间位姿求解，并在 Stack_Report 中
    记录所选路径的稳定机器可读标识符。
12. IF 第 5、第 6 或第 7 条中任一判据未满足，THEN THE Acceptance_Harness SHALL 把该次运行判定为失败，
    保留已产出的 Stack_Report，并在 Stack_Report 中记录未满足判据的稳定机器可读标识符、实测值和阈值。
