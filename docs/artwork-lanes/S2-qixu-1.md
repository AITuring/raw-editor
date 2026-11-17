# S2-qixu-1：祁序1全量摸底

- 更新：2026-11-18；版本：baseline-r1；对应任务5；执行者/L0为本次同一执行者。
- 目的：只跑原冻结基线、记录失败和能力缺口；无算法、阈值或像素预期改动。
- 起点：v1 `1da9fd44`；工作区`/Users/dp/Desktop/raw-editor/.worktrees/task-5`；分支`lane/task-5/baseline`。
- 输入：已登记全量75张，scale=0.25；manifest为`tests/acceptance/focus-stack-datasets.json`，SHA `46b4f77fa6a41852223954aa9f6fe8eb92eee3b737b465fac1ff4e05d07e6173`；选中输入清单SHA `a8512539353b33e3298ff371d443f428d008facda409e332e2b870a413e03460`。
- 冻结：`base-2026-11-12`；原始提交`c0c1e5a9037fb4f8a8f8b983802848feb667b9ba`；二进制SHA `39bfe6f2aecc8a7ebc154efb94ba4b1f1a26257c825a676a0e54e8cde9cef03b`。build.json位于`/Users/dp/Desktop/raw-editor/output/acceptance/binaries/base-2026-11-12/build.json`。
- 资源：exclusive，Rayon≤8、jobs≤6、nice≥10；该版本未验证共享线程合同，兼容规则强制独占。未知全量峰值同样须独占。
- 原始输出：`/Users/dp/Desktop/raw-editor/output/acceptance/task-5/qixu-1-baseline-r1`；机器诊断：[完整字段、全源清单与摘要](../benchmarks/qixu-1-baseline-r1-diagnostic.json)。
- runner、工具、环境与原始命令绑定在execution.json；输入和参考只读，前后核对见analysis.json。

## 1. 来源参与与排除

- 分类计数：{"excluded":4,"composited":70,"detail_only":1}，互斥合计75/75；`detail_only`已去除同时streamed的帧。
- `composited`来自Streaming日志；`singleton_tile_input`是单帧机位直接来源，未执行多帧fusion；`station_fusion_input`来自虚拟tile provenance及已完成机位fusion，只证明参与机位融合，不宣称每帧持有最终像素；`detail_only`由最终detail_repairs的donor记录证明；仅Lending但无转移记录的帧单列为detail_candidate_only。
- `prepared_station_input`只证明解码/分组参与，运行未出图时不声明最终像素归属。`excluded`须逐项结合native_reason_lines解释；`unresolved`不能补成参与或排除。全部源图的绝对路径、输入SHA、排序/机位、解码及原生日志证据保存在机器诊断的sources数组。

| 源图文件名（完整路径见JSON） | 分类 | 机位 | 排序中 | 原始原因 |
| --- | --- | --- | --- | --- |
| DSC_3842.NEF | excluded | unavailable（该报告未记录） | false | 3 image(s) could not be aligned with the selected set: DSC_3842.NEF, DSC_3910.NEF, DSC_3911.NEF. |
| DSC_3843.NEF | composited | 0 | true | native log: Streaming projected region |
| DSC_3844.NEF | composited | 0 | true | native log: Streaming projected region |
| DSC_3845.NEF | composited | 9 | true | native log: Streaming projected region |
| DSC_3846.NEF | composited | 9 | true | native log: Streaming projected region |
| DSC_3847.NEF | composited | 19 | true | - Focus station relation rejected 'DSC_3858.NEF'<->'DSC_3847.NEF': scale 1.021, overlap 48.0%, NCC 0.321, edges 0.180/0.516, samples 1230 |
| DSC_3848.NEF | composited | 19 | true | native log: Streaming projected region |
| DSC_3849.NEF | composited | 19 | true | - Focus station relation rejected 'DSC_3849.NEF'<->'DSC_3857.NEF': scale 1.027, overlap 48.0%, NCC 0.398, edges 0.286/0.621, samples 974；- Focus station relation rejected 'DSC_3849.NEF'<->'DSC_3854.NEF': scale 1.092, overlap 24.0%, NCC 0.418, edges 0.308/0.792, samples 539 |
| DSC_3850.NEF | composited | 32 | true | - Focus station relation rejected 'DSC_3850.NEF'<->'DSC_3854.NEF': scale 1.002, overlap 48.0%, NCC 0.444, edges 0.481/0.826, samples 1164 |
| DSC_3851.NEF | composited | 32 | true | native log: Streaming projected region |
| DSC_3852.NEF | composited | 36 | true | native log: Streaming projected region |
| DSC_3853.NEF | composited | 36 | true | - Focus station relation rejected 'DSC_3853.NEF'<->'DSC_3857.NEF': scale 0.902, overlap 24.0%, NCC 0.314, edges 0.483/0.731, samples 559 |
| DSC_3854.NEF | composited | 35 | true | - Focus station relation rejected 'DSC_3849.NEF'<->'DSC_3854.NEF': scale 1.092, overlap 24.0%, NCC 0.418, edges 0.308/0.792, samples 539；- Focus station relation rejected 'DSC_3850.NEF'<->'DSC_3854.NEF': scale 1.002, overlap 48.0%, NCC 0.444, edges 0.481/0.826, samples 1164 |
| DSC_3855.NEF | composited | 35 | true | - Focus geometry retry candidate rejected edge 'DSC_3855.NEF'<->'DSC_3856.NEF': scale 0.944, overlap 36.0%, NCC 0.416, edges 0.306/0.628, samples 1334；- Focus station relation rejected 'DSC_3864.NEF'<->'DSC_3855.NEF': scale 0.966, overlap 24.0%, NCC 0.397, edges 0.182/0.332, samples 639 |
| DSC_3856.NEF | composited | 25 | true | - Focus geometry retry candidate rejected edge 'DSC_3855.NEF'<->'DSC_3856.NEF': scale 0.944, overlap 36.0%, NCC 0.416, edges 0.306/0.628, samples 1334；- Focus station relation rejected 'DSC_3864.NEF'<->'DSC_3856.NEF': scale 0.910, overlap 28.0%, NCC 0.272, edges 0.175/0.503, samples 1016 |
| DSC_3857.NEF | composited | 25 | true | - Focus station relation rejected 'DSC_3861.NEF'<->'DSC_3857.NEF': scale 0.994, overlap 12.0%, NCC 0.386, edges 0.191/0.032, samples 201；- Focus station relation rejected 'DSC_3862.NEF'<->'DSC_3857.NEF': scale 0.946, overlap 36.0%, NCC 0.341, edges 0.182/0.614, samples 711；- Focus station relation rejected 'DSC_3849.NEF'<->'DSC_3857.NEF': scale 1.027, overlap 48.0%, NCC 0.398, edges 0.286/0.621, samples 974；- Focus station relation rejected 'DSC_3853.NEF'<->'DSC_3857.NEF': scale 0.902, overlap 24.0%, NCC 0.314, edges 0.483/0.731, samples 559 |
| DSC_3858.NEF | composited | 17 | true | - Focus geometry retry candidate rejected edge 'DSC_3858.NEF'<->'DSC_3859.NEF': scale 0.983, overlap 24.0%, NCC 0.022, edges -0.034/-0.022, samples 1080；- Focus station relation rejected 'DSC_3858.NEF'<->'DSC_3847.NEF': scale 1.021, overlap 48.0%, NCC 0.321, edges 0.180/0.516, samples 1230 |
| DSC_3859.NEF | composited | 1 | true | - Focus geometry retry candidate rejected edge 'DSC_3858.NEF'<->'DSC_3859.NEF': scale 0.983, overlap 24.0%, NCC 0.022, edges -0.034/-0.022, samples 1080 |
| DSC_3860.NEF | composited | 3 | true | - Focus station relation rejected 'DSC_3860.NEF'<->'DSC_3868.NEF': scale 1.076, overlap 24.0%, NCC 0.221, edges 0.120/0.052, samples 407 |
| DSC_3861.NEF | composited | 3 | true | - Focus station relation rejected 'DSC_3864.NEF'<->'DSC_3861.NEF': scale 0.912, overlap 12.0%, NCC 0.244, edges 0.066/0.496, samples 274；- Focus station relation rejected 'DSC_3861.NEF'<->'DSC_3857.NEF': scale 0.994, overlap 12.0%, NCC 0.386, edges 0.191/0.032, samples 201 |
| DSC_3862.NEF | composited | 13 | true | - Focus station relation rejected 'DSC_3862.NEF'<->'DSC_3857.NEF': scale 0.946, overlap 36.0%, NCC 0.341, edges 0.182/0.614, samples 711；- Focus station relation rejected 'DSC_3865.NEF'<->'DSC_3862.NEF': scale 0.943, overlap 24.0%, NCC 0.325, edges 0.161/0.550, samples 720 |
| DSC_3863.NEF | composited | 13 | true | native log: Streaming projected region |
| DSC_3864.NEF | composited | 24 | true | - Focus station relation rejected 'DSC_3864.NEF'<->'DSC_3861.NEF': scale 0.912, overlap 12.0%, NCC 0.244, edges 0.066/0.496, samples 274；- Focus station relation rejected 'DSC_3864.NEF'<->'DSC_3856.NEF': scale 0.910, overlap 28.0%, NCC 0.272, edges 0.175/0.503, samples 1016；- Focus station relation rejected 'DSC_3864.NEF'<->'DSC_3855.NEF': scale 0.966, overlap 24.0%, NCC 0.397, edges 0.182/0.332, samples 639 |
| DSC_3865.NEF | composited | 31 | true | - Focus station relation rejected 'DSC_3865.NEF'<->'DSC_3862.NEF': scale 0.943, overlap 24.0%, NCC 0.325, edges 0.161/0.550, samples 720 |
| DSC_3866.NEF | composited | 31 | true | - Focus geometry retry candidate rejected edge 'DSC_3866.NEF'<->'DSC_3867.NEF': scale 0.960, overlap 16.0%, NCC -0.191, edges -0.099/0.152, samples 206 |
| DSC_3867.NEF | composited | 15 | true | - Focus geometry retry candidate rejected edge 'DSC_3866.NEF'<->'DSC_3867.NEF': scale 0.960, overlap 16.0%, NCC -0.191, edges -0.099/0.152, samples 206 |
| DSC_3868.NEF | composited | 15 | true | - Focus geometry retry candidate rejected edge 'DSC_3868.NEF'<->'DSC_3869.NEF': scale 0.965, overlap 48.0%, NCC 0.379, edges -0.013/0.110, samples 1232；- Focus station relation rejected 'DSC_3860.NEF'<->'DSC_3868.NEF': scale 1.076, overlap 24.0%, NCC 0.221, edges 0.120/0.052, samples 407；- Focus station relation rejected 'DSC_3868.NEF'<->'DSC_3878.NEF': scale 1.130, overlap 48.0%, NCC 0.374, edges 0.209/0.648, samples 1126；- Focus station relation rejected 'DSC_3868.NEF'<->'DSC_3873.NEF': scale 1.187, overlap 12.0%, NCC 0.372, edges 0.085/0.303, samples 360 |
| DSC_3869.NEF | composited | 7 | true | - Focus geometry retry candidate rejected edge 'DSC_3868.NEF'<->'DSC_3869.NEF': scale 0.965, overlap 48.0%, NCC 0.379, edges -0.013/0.110, samples 1232；- Focus geometry retry candidate rejected edge 'DSC_3869.NEF'<->'DSC_3870.NEF': scale 1.147, overlap 12.0%, NCC 0.144, edges 0.134/-0.220, samples 328；- Focus station relation rejected 'DSC_3869.NEF'<->'DSC_3882.NEF': scale 1.132, overlap 16.0%, NCC 0.395, edges 0.242/0.476, samples 324 |
| DSC_3870.NEF | composited | 26 | true | - Focus geometry retry candidate rejected edge 'DSC_3869.NEF'<->'DSC_3870.NEF': scale 1.147, overlap 12.0%, NCC 0.144, edges 0.134/-0.220, samples 328；- Focus station relation rejected 'DSC_3870.NEF'<->'DSC_3878.NEF': scale 1.020, overlap 36.0%, NCC 0.277, edges 0.245/0.667, samples 963 |
| DSC_3871.NEF | composited | 26 | true | native log: Streaming projected region |
| DSC_3872.NEF | composited | 34 | true | native log: Streaming projected region |
| DSC_3873.NEF | composited | 34 | true | - Focus station relation rejected 'DSC_3868.NEF'<->'DSC_3873.NEF': scale 1.187, overlap 12.0%, NCC 0.372, edges 0.085/0.303, samples 360；- Focus station relation rejected 'DSC_3873.NEF'<->'DSC_3878.NEF': scale 0.950, overlap 24.0%, NCC 0.290, edges 0.037/0.349, samples 330 |
| DSC_3874.NEF | composited | 28 | true | native log: Streaming projected region |
| DSC_3875.NEF | composited | 28 | true | - Focus station relation rejected 'DSC_3888.NEF'<->'DSC_3875.NEF': scale 0.959, overlap 12.0%, NCC 0.223, edges 0.013/0.196, samples 373；- Focus station relation rejected 'DSC_3875.NEF'<->'DSC_3892.NEF': scale 1.089, overlap 32.0%, NCC 0.409, edges 0.175/0.002, samples 823 |
| DSC_3876.NEF | composited | 28 | true | - Focus geometry retry candidate rejected edge 'DSC_3876.NEF'<->'DSC_3877.NEF': scale 0.941, overlap 40.0%, NCC 0.370, edges 0.115/0.421, samples 1098 |
| DSC_3877.NEF | composited | 20 | true | - Focus geometry retry candidate rejected edge 'DSC_3876.NEF'<->'DSC_3877.NEF': scale 0.941, overlap 40.0%, NCC 0.370, edges 0.115/0.421, samples 1098 |
| DSC_3878.NEF | composited | 20 | true | - Focus geometry retry candidate rejected edge 'DSC_3878.NEF'<->'DSC_3879.NEF': scale 0.969, overlap 36.0%, NCC 0.423, edges 0.095/0.549, samples 1638；- Focus station relation rejected 'DSC_3868.NEF'<->'DSC_3878.NEF': scale 1.130, overlap 48.0%, NCC 0.374, edges 0.209/0.648, samples 1126；- Focus station relation rejected 'DSC_3878.NEF'<->'DSC_3889.NEF': scale 1.129, overlap 32.0%, NCC 0.336, edges 0.139/0.411, samples 531；- Focus station relation rejected 'DSC_3870.NEF'<->'DSC_3878.NEF': scale 1.020, overlap 36.0%, NCC 0.277, edges 0.245/0.667, samples 963；- Focus station relation rejected 'DSC_3873.NEF'<->'DSC_3878.NEF': scale 0.950, overlap 24.0%, NCC 0.290, edges 0.037/0.349, samples 330 |
| DSC_3879.NEF | excluded | 12 | true | - Focus geometry retry candidate rejected edge 'DSC_3878.NEF'<->'DSC_3879.NEF': scale 0.969, overlap 36.0%, NCC 0.423, edges 0.095/0.549, samples 1638；- Focus station relation rejected 'DSC_3882.NEF'<->'DSC_3879.NEF': scale 1.021, overlap 24.0%, NCC 0.270, edges 0.329/0.771, samples 631；- Focus station relation rejected 'DSC_3879.NEF'<->'DSC_3905.NEF': scale 1.337, overlap 16.0%, NCC 0.263, edges 0.223/0.816, samples 508；- Focus station relation rejected 'DSC_3886.NEF'<->'DSC_3879.NEF': scale 0.928, overlap 32.0%, NCC 0.283, edges 0.168/0.666, samples 800；- Focus scan: leaving out 2 mostly soft frame(s): DSC_3879.NEF (60% of its cells soft), DSC_3907.NEF (46% of its cells soft) |
| DSC_3880.NEF | composited | 8 | true | - Focus station relation rejected 'DSC_3880.NEF'<->'DSC_3905.NEF': scale 1.413, overlap 24.0%, NCC 0.331, edges 0.215/0.711, samples 971；- Focus station relation rejected 'DSC_3902.NEF'<->'DSC_3880.NEF': scale 0.921, overlap 16.0%, NCC 0.414, edges 0.159/0.828, samples 364 |
| DSC_3881.NEF | composited | 8 | true | - Focus station relation rejected 'DSC_3907.NEF'<->'DSC_3881.NEF': scale 0.740, overlap 24.0%, NCC 0.337, edges 0.165/0.443, samples 855 |
| DSC_3882.NEF | composited | 2 | true | - Focus station relation rejected 'DSC_3882.NEF'<->'DSC_3905.NEF': scale 1.351, overlap 36.0%, NCC 0.922, edges 0.424/0.839, samples 2160；- Focus station relation rejected 'DSC_3869.NEF'<->'DSC_3882.NEF': scale 1.132, overlap 16.0%, NCC 0.395, edges 0.242/0.476, samples 324；- Focus station relation rejected 'DSC_3882.NEF'<->'DSC_3904.NEF': scale 1.479, overlap 36.0%, NCC 0.396, edges 0.314/0.805, samples 2010；- Focus station relation rejected 'DSC_3882.NEF'<->'DSC_3879.NEF': scale 1.021, overlap 24.0%, NCC 0.270, edges 0.329/0.771, samples 631 |
| DSC_3883.NEF | composited | 2 | true | - Focus station relation rejected 'DSC_3883.NEF'<->'DSC_3908.NEF': scale 1.305, overlap 36.0%, NCC 0.895, edges 0.380/0.820, samples 2160；- Focus station relation rejected 'DSC_3883.NEF'<->'DSC_3912.NEF': scale 1.587, overlap 8.0%, NCC 0.603, edges 0.165/-0.401, samples 300 |
| DSC_3884.NEF | composited | 2 | true | - Focus station relation rejected 'DSC_3884.NEF'<->'DSC_3897.NEF': scale 1.178, overlap 12.0%, NCC 0.424, edges 0.424/0.796, samples 168；- Focus station relation rejected 'DSC_3884.NEF'<->'DSC_3889.NEF': scale 1.157, overlap 16.0%, NCC 0.332, edges 0.106/0.770, samples 148 |
| DSC_3885.NEF | composited | 2 | true | - Focus geometry retry candidate rejected edge 'DSC_3885.NEF'<->'DSC_3886.NEF': scale 1.098, overlap 32.0%, NCC 0.423, edges 0.348/0.778, samples 1089；- Focus station relation rejected 'DSC_3885.NEF'<->'DSC_3903.NEF': scale 1.547, overlap 24.0%, NCC 0.416, edges 0.105/0.703, samples 1053 |
| DSC_3886.NEF | composited | 16 | true | - Focus geometry retry candidate rejected edge 'DSC_3885.NEF'<->'DSC_3886.NEF': scale 1.098, overlap 32.0%, NCC 0.423, edges 0.348/0.778, samples 1089；- Focus station relation rejected 'DSC_3886.NEF'<->'DSC_3908.NEF': scale 1.194, overlap 32.0%, NCC 0.647, edges 0.281/0.700, samples 1045；- Focus station relation rejected 'DSC_3886.NEF'<->'DSC_3906.NEF': scale 1.242, overlap 24.0%, NCC 0.575, edges 0.219/0.784, samples 1076；- Focus station relation rejected 'DSC_3886.NEF'<->'DSC_3902.NEF': scale 0.971, overlap 36.0%, NCC 0.423, edges 0.171/0.766, samples 795；- Focus station relation rejected 'DSC_3886.NEF'<->'DSC_3879.NEF': scale 0.928, overlap 32.0%, NCC 0.283, edges 0.168/0.666, samples 800；- Focus station relation rejected 'DSC_3886.NEF'<->'DSC_3903.NEF': scale 1.410, overlap 36.0%, NCC 0.443, edges 0.365/0.769, samples 1902；- Focus station relation rejected 'DSC_3886.NEF'<->'DSC_3913.NEF': scale 1.453, overlap 24.0%, NCC 0.523, edges 0.275/0.611, samples 2040；- Focus station relation rejected 'DSC_3886.NEF'<->'DSC_3896.NEF': scale 1.080, overlap 36.0%, NCC 0.395, edges 0.395/0.824, samples 1024 |
| DSC_3887.NEF | composited | 16 | true | native log: Streaming projected region |
| DSC_3888.NEF | composited | 23 | true | - Focus station relation rejected 'DSC_3888.NEF'<->'DSC_3903.NEF': scale 1.331, overlap 24.0%, NCC 0.409, edges 0.092/0.679, samples 1067；- Focus station relation rejected 'DSC_3888.NEF'<->'DSC_3875.NEF': scale 0.959, overlap 12.0%, NCC 0.223, edges 0.013/0.196, samples 373 |
| DSC_3889.NEF | composited | 23 | true | - Focus station relation rejected 'DSC_3884.NEF'<->'DSC_3889.NEF': scale 1.157, overlap 16.0%, NCC 0.332, edges 0.106/0.770, samples 148；- Focus station relation rejected 'DSC_3889.NEF'<->'DSC_3899.NEF': scale 0.982, overlap 36.0%, NCC 0.446, edges 0.292/0.823, samples 943；- Focus station relation rejected 'DSC_3878.NEF'<->'DSC_3889.NEF': scale 1.129, overlap 32.0%, NCC 0.336, edges 0.139/0.411, samples 531；- Focus station relation rejected 'DSC_3889.NEF'<->'DSC_3913.NEF': scale 1.368, overlap 36.0%, NCC 0.742, edges 0.247/0.643, samples 2160；- Focus station relation rejected 'DSC_3894.NEF'<->'DSC_3889.NEF': scale 0.931, overlap 36.0%, NCC 0.391, edges 0.358/0.749, samples 966 |
| DSC_3890.NEF | composited | 30 | true | - Focus station relation rejected 'DSC_3890.NEF'<->'DSC_3915.NEF': scale 1.564, overlap 20.0%, NCC 0.298, edges 0.172/0.554, samples 1764 |
| DSC_3891.NEF | composited | 30 | true | native log: Streaming projected region |
| DSC_3892.NEF | composited | 27 | true | - Focus station relation rejected 'DSC_3875.NEF'<->'DSC_3892.NEF': scale 1.089, overlap 32.0%, NCC 0.409, edges 0.175/0.002, samples 823 |
| DSC_3893.NEF | composited | 27 | true | native log: Streaming projected region |
| DSC_3894.NEF | composited | 29 | true | - Focus station relation rejected 'DSC_3894.NEF'<->'DSC_3903.NEF': scale 1.234, overlap 12.0%, NCC 0.638, edges 0.099/0.263, samples 240；- Focus station relation rejected 'DSC_3894.NEF'<->'DSC_3889.NEF': scale 0.931, overlap 36.0%, NCC 0.391, edges 0.358/0.749, samples 966 |
| DSC_3895.NEF | composited | 29 | true | - Focus station relation rejected 'DSC_3895.NEF'<->'DSC_3912.NEF': scale 1.276, overlap 24.0%, NCC 0.251, edges 0.108/0.346, samples 923 |
| DSC_3896.NEF | composited | 22 | true | - Focus station relation rejected 'DSC_3886.NEF'<->'DSC_3896.NEF': scale 1.080, overlap 36.0%, NCC 0.395, edges 0.395/0.824, samples 1024；- Focus station relation rejected 'DSC_3913.NEF'<->'DSC_3896.NEF': scale 0.739, overlap 32.0%, NCC 0.379, edges 0.317/0.698, samples 1072 |
| DSC_3897.NEF | composited | 22 | true | - Focus station relation rejected 'DSC_3884.NEF'<->'DSC_3897.NEF': scale 1.178, overlap 12.0%, NCC 0.424, edges 0.424/0.796, samples 168；- Focus station relation rejected 'DSC_3897.NEF'<->'DSC_3903.NEF': scale 1.305, overlap 32.0%, NCC 0.816, edges 0.213/0.513, samples 1500 |
| DSC_3898.NEF | composited | 18 | true | - Focus station relation rejected 'DSC_3898.NEF'<->'DSC_3905.NEF': scale 1.193, overlap 24.0%, NCC 0.338, edges 0.206/0.769, samples 765；- Focus station relation rejected 'DSC_3898.NEF'<->'DSC_3904.NEF': scale 1.304, overlap 32.0%, NCC 0.514, edges 0.176/0.771, samples 1560；- Focus station relation rejected 'DSC_3898.NEF'<->'DSC_3903.NEF': scale 1.363, overlap 36.0%, NCC 0.930, edges 0.319/0.571, samples 2160 |
| DSC_3899.NEF | composited | 18 | true | - Focus station relation rejected 'DSC_3909.NEF'<->'DSC_3899.NEF': scale 0.842, overlap 24.0%, NCC 0.244, edges 0.081/0.483, samples 397；- Focus station relation rejected 'DSC_3913.NEF'<->'DSC_3899.NEF': scale 0.711, overlap 36.0%, NCC 0.489, edges 0.377/0.591, samples 949；- Focus station relation rejected 'DSC_3889.NEF'<->'DSC_3899.NEF': scale 0.982, overlap 36.0%, NCC 0.446, edges 0.292/0.823, samples 943 |
| DSC_3900.NEF | composited | 11 | true | - Focus station relation rejected 'DSC_3905.NEF'<->'DSC_3900.NEF': scale 0.815, overlap 36.0%, NCC 0.521, edges 0.288/0.656, samples 877；- Focus station relation rejected 'DSC_3903.NEF'<->'DSC_3900.NEF': scale 0.710, overlap 36.0%, NCC 0.363, edges 0.126/0.466, samples 987；- Focus station relation rejected 'DSC_3912.NEF'<->'DSC_3900.NEF': scale 0.690, overlap 24.0%, NCC 0.438, edges 0.029/0.469, samples 464 |
| DSC_3901.NEF | composited | 11 | true | - Focus station relation rejected 'DSC_3907.NEF'<->'DSC_3901.NEF': scale 0.833, overlap 36.0%, NCC 0.570, edges 0.156/0.707, samples 712 |
| DSC_3902.NEF | composited | 6 | true | - Focus geometry retry candidate rejected edge 'DSC_3902.NEF'<->'DSC_3903.NEF': scale 1.457, overlap 24.0%, NCC 0.797, edges 0.030/0.256, samples 1320；- Focus station relation rejected 'DSC_3902.NEF'<->'DSC_3907.NEF': scale 1.238, overlap 36.0%, NCC 0.710, edges 0.334/0.754, samples 1476；- Focus station relation rejected 'DSC_3902.NEF'<->'DSC_3906.NEF': scale 1.288, overlap 36.0%, NCC 0.483, edges 0.325/0.796, samples 1814；- Focus station relation rejected 'DSC_3902.NEF'<->'DSC_3880.NEF': scale 0.921, overlap 16.0%, NCC 0.414, edges 0.159/0.828, samples 364；- Focus station relation rejected 'DSC_3902.NEF'<->'DSC_3904.NEF': scale 1.394, overlap 16.0%, NCC 0.847, edges 0.369/0.803, samples 2160；- Focus station relation rejected 'DSC_3886.NEF'<->'DSC_3902.NEF': scale 0.971, overlap 36.0%, NCC 0.423, edges 0.171/0.766, samples 795 |
| DSC_3903.NEF | composited | 14 | true | - Focus geometry retry candidate rejected edge 'DSC_3902.NEF'<->'DSC_3903.NEF': scale 1.457, overlap 24.0%, NCC 0.797, edges 0.030/0.256, samples 1320；- Focus geometry retry candidate rejected edge 'DSC_3903.NEF'<->'DSC_3904.NEF': scale 0.957, overlap 48.0%, NCC 0.318, edges 0.035/0.584, samples 1134；- Focus station relation rejected 'DSC_3885.NEF'<->'DSC_3903.NEF': scale 1.547, overlap 24.0%, NCC 0.416, edges 0.105/0.703, samples 1053；- Focus station relation rejected 'DSC_3909.NEF'<->'DSC_3903.NEF': scale 1.149, overlap 24.0%, NCC 0.263, edges -0.040/0.409, samples 471；- Focus station relation rejected 'DSC_3903.NEF'<->'DSC_3900.NEF': scale 0.710, overlap 36.0%, NCC 0.363, edges 0.126/0.466, samples 987；- Focus station relation rejected 'DSC_3886.NEF'<->'DSC_3903.NEF': scale 1.410, overlap 36.0%, NCC 0.443, edges 0.365/0.769, samples 1902；- Focus station relation rejected 'DSC_3898.NEF'<->'DSC_3903.NEF': scale 1.363, overlap 36.0%, NCC 0.930, edges 0.319/0.571, samples 2160；- Focus station relation rejected 'DSC_3903.NEF'<->'DSC_3913.NEF': scale 1.033, overlap 48.0%, NCC 0.311, edges 0.288/0.451, samples 1271；- Focus station relation rejected 'DSC_3897.NEF'<->'DSC_3903.NEF': scale 1.305, overlap 32.0%, NCC 0.816, edges 0.213/0.513, samples 1500；- Focus station relation rejected 'DSC_3888.NEF'<->'DSC_3903.NEF': scale 1.331, overlap 24.0%, NCC 0.409, edges 0.092/0.679, samples 1067；- Focus station relation rejected 'DSC_3894.NEF'<->'DSC_3903.NEF': scale 1.234, overlap 12.0%, NCC 0.638, edges 0.099/0.263, samples 240 |
| DSC_3904.NEF | composited | 10 | true | - Focus geometry retry candidate rejected edge 'DSC_3903.NEF'<->'DSC_3904.NEF': scale 0.957, overlap 48.0%, NCC 0.318, edges 0.035/0.584, samples 1134；- Focus geometry retry candidate rejected edge 'DSC_3904.NEF'<->'DSC_3905.NEF': scale 0.917, overlap 56.0%, NCC 0.376, edges 0.305/0.629, samples 1314；- Focus station relation rejected 'DSC_3882.NEF'<->'DSC_3904.NEF': scale 1.479, overlap 36.0%, NCC 0.396, edges 0.314/0.805, samples 2010；- Focus station relation rejected 'DSC_3909.NEF'<->'DSC_3904.NEF': scale 1.099, overlap 44.0%, NCC 0.425, edges 0.138/0.558, samples 1261；- Focus station relation rejected 'DSC_3906.NEF'<->'DSC_3904.NEF': scale 1.083, overlap 64.0%, NCC 0.435, edges 0.313/0.608, samples 1494；- Focus station relation rejected 'DSC_3902.NEF'<->'DSC_3904.NEF': scale 1.394, overlap 16.0%, NCC 0.847, edges 0.369/0.803, samples 2160；- Focus station relation rejected 'DSC_3898.NEF'<->'DSC_3904.NEF': scale 1.304, overlap 32.0%, NCC 0.514, edges 0.176/0.771, samples 1560 |
| DSC_3905.NEF | composited | 5 | true | - Focus geometry retry candidate rejected edge 'DSC_3904.NEF'<->'DSC_3905.NEF': scale 0.917, overlap 56.0%, NCC 0.376, edges 0.305/0.629, samples 1314；- Focus station relation rejected 'DSC_3882.NEF'<->'DSC_3905.NEF': scale 1.351, overlap 36.0%, NCC 0.922, edges 0.424/0.839, samples 2160；- Focus station relation rejected 'DSC_3880.NEF'<->'DSC_3905.NEF': scale 1.413, overlap 24.0%, NCC 0.331, edges 0.215/0.711, samples 971；- Focus station relation rejected 'DSC_3905.NEF'<->'DSC_3900.NEF': scale 0.815, overlap 36.0%, NCC 0.521, edges 0.288/0.656, samples 877；- Focus station relation rejected 'DSC_3879.NEF'<->'DSC_3905.NEF': scale 1.337, overlap 16.0%, NCC 0.263, edges 0.223/0.816, samples 508；- Focus station relation rejected 'DSC_3898.NEF'<->'DSC_3905.NEF': scale 1.193, overlap 24.0%, NCC 0.338, edges 0.206/0.769, samples 765 |
| DSC_3906.NEF | composited | 5 | true | - Focus station relation rejected 'DSC_3902.NEF'<->'DSC_3906.NEF': scale 1.288, overlap 36.0%, NCC 0.483, edges 0.325/0.796, samples 1814；- Focus station relation rejected 'DSC_3906.NEF'<->'DSC_3904.NEF': scale 1.083, overlap 64.0%, NCC 0.435, edges 0.313/0.608, samples 1494；- Focus station relation rejected 'DSC_3886.NEF'<->'DSC_3906.NEF': scale 1.242, overlap 24.0%, NCC 0.575, edges 0.219/0.784, samples 1076 |
| DSC_3907.NEF | detail_only | 4 | true | - Focus station relation rejected 'DSC_3902.NEF'<->'DSC_3907.NEF': scale 1.238, overlap 36.0%, NCC 0.710, edges 0.334/0.754, samples 1476；- Focus station relation rejected 'DSC_3907.NEF'<->'DSC_3881.NEF': scale 0.740, overlap 24.0%, NCC 0.337, edges 0.165/0.443, samples 855；- Focus station relation rejected 'DSC_3907.NEF'<->'DSC_3901.NEF': scale 0.833, overlap 36.0%, NCC 0.570, edges 0.156/0.707, samples 712；- Focus scan: leaving out 2 mostly soft frame(s): DSC_3879.NEF (60% of its cells soft), DSC_3907.NEF (46% of its cells soft) |
| DSC_3908.NEF | composited | 4 | true | - Focus station relation rejected 'DSC_3883.NEF'<->'DSC_3908.NEF': scale 1.305, overlap 36.0%, NCC 0.895, edges 0.380/0.820, samples 2160；- Focus station relation rejected 'DSC_3886.NEF'<->'DSC_3908.NEF': scale 1.194, overlap 32.0%, NCC 0.647, edges 0.281/0.700, samples 1045 |
| DSC_3909.NEF | composited | 4 | true | - Focus geometry retry candidate rejected edge 'DSC_3909.NEF'<->'DSC_3912.NEF': scale 1.185, overlap 8.0%, NCC -0.062, edges -0.046/0.010, samples 693；- Focus station relation rejected 'DSC_3909.NEF'<->'DSC_3904.NEF': scale 1.099, overlap 44.0%, NCC 0.425, edges 0.138/0.558, samples 1261；- Focus station relation rejected 'DSC_3909.NEF'<->'DSC_3903.NEF': scale 1.149, overlap 24.0%, NCC 0.263, edges -0.040/0.409, samples 471；- Focus station relation rejected 'DSC_3909.NEF'<->'DSC_3899.NEF': scale 0.842, overlap 24.0%, NCC 0.244, edges 0.081/0.483, samples 397 |
| DSC_3910.NEF | excluded | unavailable（该报告未记录） | false | 3 image(s) could not be aligned with the selected set: DSC_3842.NEF, DSC_3910.NEF, DSC_3911.NEF. |
| DSC_3911.NEF | excluded | unavailable（该报告未记录） | false | 3 image(s) could not be aligned with the selected set: DSC_3842.NEF, DSC_3910.NEF, DSC_3911.NEF. |
| DSC_3912.NEF | composited | 21 | true | - Focus geometry retry candidate rejected edge 'DSC_3909.NEF'<->'DSC_3912.NEF': scale 1.185, overlap 8.0%, NCC -0.062, edges -0.046/0.010, samples 693；- Focus station relation rejected 'DSC_3883.NEF'<->'DSC_3912.NEF': scale 1.587, overlap 8.0%, NCC 0.603, edges 0.165/-0.401, samples 300；- Focus station relation rejected 'DSC_3912.NEF'<->'DSC_3900.NEF': scale 0.690, overlap 24.0%, NCC 0.438, edges 0.029/0.469, samples 464；- Focus station relation rejected 'DSC_3895.NEF'<->'DSC_3912.NEF': scale 1.276, overlap 24.0%, NCC 0.251, edges 0.108/0.346, samples 923 |
| DSC_3913.NEF | composited | 21 | true | - Focus geometry retry candidate rejected edge 'DSC_3913.NEF'<->'DSC_3914.NEF': scale 1.156, overlap 36.0%, NCC 0.208, edges -0.003/-0.069, samples 1064；- Focus station relation rejected 'DSC_3903.NEF'<->'DSC_3913.NEF': scale 1.033, overlap 48.0%, NCC 0.311, edges 0.288/0.451, samples 1271；- Focus station relation rejected 'DSC_3886.NEF'<->'DSC_3913.NEF': scale 1.453, overlap 24.0%, NCC 0.523, edges 0.275/0.611, samples 2040；- Focus station relation rejected 'DSC_3913.NEF'<->'DSC_3899.NEF': scale 0.711, overlap 36.0%, NCC 0.489, edges 0.377/0.591, samples 949；- Focus station relation rejected 'DSC_3913.NEF'<->'DSC_3896.NEF': scale 0.739, overlap 32.0%, NCC 0.379, edges 0.317/0.698, samples 1072；- Focus station relation rejected 'DSC_3889.NEF'<->'DSC_3913.NEF': scale 1.368, overlap 36.0%, NCC 0.742, edges 0.247/0.643, samples 2160 |
| DSC_3914.NEF | composited | unavailable（该报告未记录） | true | - Focus geometry retry candidate rejected edge 'DSC_3913.NEF'<->'DSC_3914.NEF': scale 1.156, overlap 36.0%, NCC 0.208, edges -0.003/-0.069, samples 1064；- Focus geometry retry candidate rejected edge 'DSC_3914.NEF'<->'DSC_3915.NEF': scale 1.068, overlap 24.0%, NCC 0.122, edges 0.017/0.180, samples 736 |
| DSC_3915.NEF | composited | 33 | true | - Focus geometry retry candidate rejected edge 'DSC_3914.NEF'<->'DSC_3915.NEF': scale 1.068, overlap 24.0%, NCC 0.122, edges 0.017/0.180, samples 736；- Focus station relation rejected 'DSC_3890.NEF'<->'DSC_3915.NEF': scale 1.564, overlap 20.0%, NCC 0.298, edges 0.172/0.554, samples 1764 |
| DSC_3916.NEF | composited | 33 | true | native log: Streaming projected region |

## 2. 实际路径、机位与停止阶段

- selected_path：`legacy_single_layer_mosaic`；station数：37；路径配置/回退：{"configured_path":null,"selected_path":"legacy_single_layer_mosaic","use_virtual_tiles":false,"fallback_reason":"isolated_source"}。
- isolated：[{"path":"/Users/dp/Downloads/书画拼图/祁序1/DSC_3914.NEF","reason":"grouping_no_overlap_evidence"}]；undecodable：[]；source_scan：{"singleton_station_count":8,"multi_frame_station_count":29,"singleton_ratio":0.21621621621621623,"detected":false,"isolated_source_count":1}。
- closure："not_run"；连通/分量：{"components":0,"component_members":[],"component_member_counts":[]}。
- 输出密度/输出像素门：{"status":"not_run","factor":1.0,"pixel_count_multiplier":1.0,"canvas_pixels":0,"output_plane_lower_bound_bytes":0,"target_minimum_scale":1.0,"probe_step_px":0,"sampling_contract":"","source_count":0,"sample_count":0,"invalid_probe_count":0,"boundary_probe_count":0,"worst_station_index":null,"worst_source_id":null,"worst_station_point":null,"direct_pair_p95_output_px":null,"residual_median_output_px":null,"max_corner_correction_output_px":null,"minimum_scale":null,"p01_scale":null,"median_scale":null,"bypass_reason":null}。
- 结束：exit=1，analysis=`rejected`；最后完成/首次失败阶段：legacy流式合成、30处细节转移/替换、来源归属、QG、25% RGBA16 TIFF与预览均完成；最终degradation=rejected，QG fail。正式runner另对未声明排除/70条Streaming计数拒绝，未修改预期。
- 原始失败：["the Stack_Report has no verified TIFF export at the requested path","verified TIFF dimensions do not match the rendered result","the TIFF export lacks verified unassociated coverage alpha (ExtraSamples=[2])","inputs missing from the stitching order: DSC_3842.NEF, DSC_3910.NEF, DSC_3911.NEF","streaming compositor used 70/75 inputs (0 lending detail only); absent: DSC_3842.NEF, DSC_3879.NEF, DSC_3907.NEF, DSC_3910.NEF, DSC_3911.NEF","the image-stack dialog rejects this result (Stack_Report degradation.result is rejected)"]；degradation及native_errors完整保存在JSON，不用TIFF存在代替成功。
- QG：{"verdict":"fail","roi_count":8,"criteria":{"local_scale_median":"not_applicable","local_scale_pixel_ratio":"fail","effective_pixel_count":"pass","mtf50_normalized":"not_applicable","gradient_energy_normalized":"pass","noise_sigma_ratio":"pass","roi_delta_e00":"pass","boundary_stroke_alignment":"fail","owner_sharpness_coverage":"fail"},"non_pass":["local_scale_median:not_applicable","local_scale_pixel_ratio:fail","mtf50_normalized:not_applicable","boundary_stroke_alignment:fail","owner_sharpness_coverage:fail"],"unresolved_pixels":0,"detail_repairs":30}。
- 独立tone状态：{"status":"not_run","boundary_delta_e":{"max":0.0,"threshold":1.5,"violations":[]},"correction_model":"station_affine","geometry_active_relation_count":0}。

## 3. N4.5失败链

以下数值直接来自本次报告。候选record、最终relation和hypothesis理由发生次数分母不同；同一pair可能有多次尝试，不直接相加。legacy路径空的station关系数组不代表注册无失败；未观测proposal/shared gate阶段标为unavailable。

```json
{
  "selected_path": "legacy_single_layer_mosaic",
  "candidates_recorded": 0,
  "model_fitted_records": 0,
  "accepted_relations": 0,
  "rejected_count": 0,
  "unmeasured_count": 0,
  "rejected_by_failure_stage": {},
  "rejected_by_reason": {},
  "candidate_stage_histogram": {},
  "hypothesis_rejection_reason_occurrences": {},
  "source_raw_pair_diagnostics_count": 0,
  "source_raw_direct_diagnostics_count": 0,
  "connectivity": {
    "components": 0,
    "component_members": [],
    "component_member_counts": []
  },
  "note": "Candidate records, final relation counts and hypothesis reason occurrences have different denominators and can repeat a pair. Empty legacy station diagnostics do not prove no registration failure. Shared-gate/proposal stages without explicit records are unavailable.",
  "proposal_count": null,
  "shared_gate_attempt_count": null,
  "station_chain_availability": "not instrumented for legacy source matching",
  "accepted_source_pairs_count": 366,
  "alignment_cache": {
    "path": "/Users/dp/Desktop/raw-editor/output/acceptance/task-5/qixu-1-baseline-r1/source-alignment.json",
    "schema": 2,
    "match_records": 366
  },
  "diagnostic_categories": {
    "no_overlap": {
      "rejected_by_failure_stage": {},
      "rejected_by_reason": {},
      "candidate_stage_histogram": {},
      "hypothesis_rejection_reason_occurrences": {}
    },
    "support": {
      "rejected_by_failure_stage": {},
      "rejected_by_reason": {},
      "candidate_stage_histogram": {},
      "hypothesis_rejection_reason_occurrences": {}
    },
    "bidirectional": {
      "rejected_by_failure_stage": {},
      "rejected_by_reason": {},
      "candidate_stage_histogram": {},
      "hypothesis_rejection_reason_occurrences": {}
    },
    "grid": {
      "rejected_by_failure_stage": {},
      "rejected_by_reason": {},
      "candidate_stage_histogram": {},
      "hypothesis_rejection_reason_occurrences": {}
    },
    "scale": {
      "rejected_by_failure_stage": {},
      "rejected_by_reason": {},
      "candidate_stage_histogram": {},
      "hypothesis_rejection_reason_occurrences": {}
    },
    "hull": {
      "rejected_by_failure_stage": {},
      "rejected_by_reason": {},
      "candidate_stage_histogram": {},
      "hypothesis_rejection_reason_occurrences": {}
    },
    "orientation": {
      "rejected_by_failure_stage": {},
      "rejected_by_reason": {},
      "candidate_stage_histogram": {},
      "hypothesis_rejection_reason_occurrences": {}
    },
    "closure": {
      "rejected_by_failure_stage": {},
      "rejected_by_reason": {},
      "candidate_stage_histogram": {},
      "hypothesis_rejection_reason_occurrences": {}
    }
  }
}
```

| 诊断类别 | 本次原始字段与统计口径 |
| --- | --- |
| no overlap | {"rejected_by_failure_stage":{},"rejected_by_reason":{},"candidate_stage_histogram":{},"hypothesis_rejection_reason_occurrences":{}} |
| support | {"rejected_by_failure_stage":{},"rejected_by_reason":{},"candidate_stage_histogram":{},"hypothesis_rejection_reason_occurrences":{}} |
| bidirectional | {"rejected_by_failure_stage":{},"rejected_by_reason":{},"candidate_stage_histogram":{},"hypothesis_rejection_reason_occurrences":{}} |
| grid | {"rejected_by_failure_stage":{},"rejected_by_reason":{},"candidate_stage_histogram":{},"hypothesis_rejection_reason_occurrences":{}} |
| scale | {"rejected_by_failure_stage":{},"rejected_by_reason":{},"candidate_stage_histogram":{},"hypothesis_rejection_reason_occurrences":{}} |
| hull | {"rejected_by_failure_stage":{},"rejected_by_reason":{},"candidate_stage_histogram":{},"hypothesis_rejection_reason_occurrences":{}} |
| orientation | {"rejected_by_failure_stage":{},"rejected_by_reason":{},"candidate_stage_histogram":{},"hypothesis_rejection_reason_occurrences":{}} |
| closure | {"rejected_by_failure_stage":{},"rejected_by_reason":{},"candidate_stage_histogram":{},"hypothesis_rejection_reason_occurrences":{}}；closure.status="not_run" |

表内空map表示“未记录该类别的相应事件”，不能据此证明没有该类问题；未经过或未观测的门级计数为unavailable。

## 4. 资源与实际输出

- 实际原生Rayon线程：8；预处理池：[6,6]。
- 峰值RSS：15116599296bytes（14.07843017578125GiB）；24GiB门：true；方法、采样数、native getrusage与报告值在resource.memory中。
- 缓存/解码计数：{"raw_decode_count":301,"native_fusion_count":0,"source_cache":{"max_bytes":536870912,"peak_bytes":511847850,"hits":21},"native_cache":{"max_bytes":1073741824,"peak_bytes":0,"hits":0}}；阶段wall：{"preflight_seconds":0.613014,"prepare_seconds":87.243534875,"matching_seconds":68.458914833,"ordering_seconds":2.537431667,"render_seconds":587.332018375,"total_seconds":746.434839666}。
- wall：938.2737998962402s；contended：false。本次独占，但不同规模/路径/scale不得据单次耗时推断性能改善。
- 磁盘由run.py在启动前检查空闲≥40GiB、lane<20GiB；精确启动余量未记录。输出/报告/日志/预览的路径和SHA均在机器诊断。
- output：{"alpha_preserved":true,"artifact_validation":"pass","artifact_validation_error":null,"bit_depth":16,"format":"TIFF","height":3155,"icc":"sRGB","path":"/Users/dp/Desktop/raw-editor/output/acceptance/task-5/qixu-1-baseline-r1/run/qixu-1-s0p25-task-5-baseline-r1.tiff","preview":{"path":"","max_roi_delta_e":0.0},"result_id":"native-fixture-56fa677f-ccb6-40af-a498-9a248cdf4bb3","tiff_extra_samples":[2],"width":6623,"written":true}；TIFF SHA：`fbf2963d1f4e56ad5d2b83dfda944f304ad3f8ab62e50db70a215c0dc3644d27`。
- 缩略预览：[{"path":"/Users/dp/Desktop/raw-editor/output/acceptance/task-5/qixu-1-baseline-r1/run/qixu-1-s0p25-task-5-baseline-r1.preview.jpg","sha256":"34b7eb071733c8ded7474f346725edd03f1c5fd270646ae94a15ce9d7dfba01e","bytes":60561,"dimensions":{"pixelWidth":1200,"pixelHeight":572}}]；六区裁片见crop_previews。均来自scale=0.25渲染，不称原生1:1裁片。

## 5. 最明显问题与初步归因

已实际查看本次1200×572预览及最终报告。主体偏暗，可见纵向明暗带/块感（预览约x=620–1000、y=100–500；对应25%输出约x=3422–5520、y=552–2758，仅为缩略观察范围，不能据此判断RAW色彩或原生细节）；外缘阶梯与黑色透明背景属于本次source-union取景，不将其自动记成内部缺洞。机器主阻断：owner_sharpness_coverage=0.9113826564746993<0.99；local_scale_pixel_ratio的8个ROI中6个测值0；boundary_stroke_alignment有3个失败采样，测值3px，原world坐标分别(-3785,977.5)、(-3896,1261.5)、(-4080,1733.5)。这些是报告world域，world_origin=(-4581,1)；减去origin可得到25%输出约(796,976.5)、(685,1260.5)、(501,1732.5)，不外推scale=1。

- QG非pass原始测量/失败坐标见机器诊断failed_criteria，保留world/owner_path原字段；无坐标时不自行转成output坐标。
- 初步区域组：流式mosaic组（任务11）；质量、来源筛除与细节donor观测一并跟进。
- 任务7（置换检验）：建议从已保存source alignment cache与accepted_source_pairs对照假边/固定门槛；legacy的station_relations空数组不能证明没有配准失败。
- 任务8（provider对照）：建议离线对照跨焦段配准和局部支持；与源图筛除、细节修复联合定位，当前不能断言仅provider可解决QG。
- 任务15（必要性能）：938.27秒正常结束，峰值14.078GiB，内存门通过；本轮无超门/无法结束证据，不启动。

## 6. 范围化结论与交付

- 当前等级：达到D1的scale=0.25诊断范围（有预览、75张均可解释）；QG失败且非原生，未达D2/D3/D4/D5。scale=0.25不能授予D2/D3；无用户D4确认，也不授予D5。
- 输入/参考保持：见analysis checks；70张主合成、DSC_3907仅供4个48×48修复区（输出x=1460/1508/1556/1604,y=526）；DSC_3842/3910/3911无法对齐而排除，DSC_3879在60%软区筛除且未出现在最终donor记录中，合计4张排除。所有75张输入前后SHA保持。正式standalone manifest未登记这些排除，runner与report_source_digests仍拒绝，原结果保持；人工诊断解释不覆盖这些门禁。
- 本卡仅覆盖本次全量清单与冻结二进制；不挪用10张子集通过结论，也不外推scale=1、其他线程数或并发场景。
- 复看：诊断结果未提交用户画质确认，队列维持0/3；预览不是已验收成片。
- L0结论：接受本轮摸底诊断与上方区域归组；修复按既有批次/依赖排队，保持全部拒绝及观测缺口，不授予原生或用户画质验收。
- 下一步：任务11归流式mosaic组，依赖任务6；优先复查局部尺度比/边界/owner覆盖及soft-frame donor记录。Node runner抓取首份中间report（75张、not_run、无导出）造成表面artifact失败，而独立analysis选中末次实测report（72张、QG fail、TIFF验证通过）；这是另一个报告选择/来源观测缺口，本任务只记录，不修工具，也不据此把QG fail改为pass。
