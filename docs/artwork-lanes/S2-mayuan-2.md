# S2-mayuan-2：马远2全量摸底

- 更新：2026-11-18；版本：baseline-r1；对应任务5；执行者/L0为本次同一执行者。
- 目的：只跑原冻结基线、记录失败和能力缺口；无算法、阈值或像素预期改动。
- 起点：v1 `1da9fd44`；工作区`/Users/dp/Desktop/raw-editor/.worktrees/task-5`；分支`lane/task-5/baseline`。
- 输入：已登记全量47张，scale=0.25；manifest为`tests/acceptance/focus-stack-datasets.json`，SHA `46b4f77fa6a41852223954aa9f6fe8eb92eee3b737b465fac1ff4e05d07e6173`；选中输入清单SHA `4e39ee193f336801a7918307307f8c9ba7d6c9490fb105678c3b4a112663294f`。
- 冻结：`base-2026-11-12`；原始提交`c0c1e5a9037fb4f8a8f8b983802848feb667b9ba`；二进制SHA `39bfe6f2aecc8a7ebc154efb94ba4b1f1a26257c825a676a0e54e8cde9cef03b`。build.json位于`/Users/dp/Desktop/raw-editor/output/acceptance/binaries/base-2026-11-12/build.json`。
- 资源：exclusive，Rayon≤8、jobs≤6、nice≥10；该版本未验证共享线程合同，兼容规则强制独占。未知全量峰值同样须独占。
- 原始输出：`/Users/dp/Desktop/raw-editor/output/acceptance/task-5/mayuan-2-baseline-r1`；机器诊断：[完整字段、全源清单与摘要](../benchmarks/mayuan-2-baseline-r1-diagnostic.json)。
- runner、工具、环境与原始命令绑定在execution.json；输入和参考只读，前后核对见analysis.json。

## 1. 来源参与与排除

- 分类计数：{"prepared_station_input":47}，互斥合计47/47；`detail_only`已去除同时streamed的帧。
- `composited`来自Streaming日志；`singleton_tile_input`是单帧机位直接来源，未执行多帧fusion；`station_fusion_input`来自虚拟tile provenance及已完成机位fusion，只证明参与机位融合，不宣称每帧持有最终像素；`detail_only`由最终detail_repairs的donor记录证明；仅Lending但无转移记录的帧单列为detail_candidate_only。
- `prepared_station_input`只证明解码/分组参与，运行未出图时不声明最终像素归属。`excluded`须逐项结合native_reason_lines解释；`unresolved`不能补成参与或排除。全部源图的绝对路径、输入SHA、排序/机位、解码及原生日志证据保存在机器诊断的sources数组。

| 源图文件名（完整路径见JSON） | 分类 | 机位 | 排序中 | 原始原因 |
| --- | --- | --- | --- | --- |
| DSC_5330.NEF | prepared_station_input | 2 | true | - Focus station relation rejected 'DSC_5436.NEF'<->'DSC_5330.NEF': scale 0.738, overlap 24.0%, NCC 0.834, edges 0.831/0.819, samples 519；- Focus station relation rejected 'DSC_5330.NEF'<->'DSC_5429.NEF': scale 1.366, overlap 12.0%, NCC 0.864, edges 0.334/0.940, samples 460 |
| DSC_5331.NEF | prepared_station_input | 4 | true | - Focus station relation rejected 'DSC_5425.NEF'<->'DSC_5331.NEF': scale 0.793, overlap 24.0%, NCC 0.801, edges 0.511/0.696, samples 865；- Focus station relation rejected 'DSC_5331.NEF'<->'DSC_5429.NEF': scale 1.365, overlap 12.0%, NCC 0.718, edges 0.651/0.862, samples 470；- Focus station relation rejected 'DSC_5331.NEF'<->'DSC_5440.NEF': scale 1.446, overlap 8.0%, NCC 0.892, edges 0.756/0.912, samples 319 |
| DSC_5332.NEF | prepared_station_input | 4 | true | - Focus station relation rejected 'DSC_5434.NEF'<->'DSC_5332.NEF': scale 0.800, overlap 40.0%, NCC 0.925, edges 0.762/0.876, samples 1116；- Focus station relation rejected 'DSC_5422.NEF'<->'DSC_5332.NEF': scale 0.855, overlap 36.0%, NCC 0.868, edges 0.455/0.728, samples 1851；- Focus station relation rejected 'DSC_5432.NEF'<->'DSC_5332.NEF': scale 0.778, overlap 36.0%, NCC 0.719, edges 0.667/0.854, samples 971；- Focus station relation rejected 'DSC_5436.NEF'<->'DSC_5332.NEF': scale 0.742, overlap 16.0%, NCC 0.768, edges 0.801/0.780, samples 522；- Focus station relation rejected 'DSC_5427.NEF'<->'DSC_5332.NEF': scale 0.755, overlap 12.0%, NCC 0.576, edges 0.389/0.589, samples 244；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5332.NEF': Local, 2917 inliers of 4325 control points, coverage 0.608, rejected 0.326, median symmetric error 0.26px, control point spacing 23.7px |
| DSC_5333.NEF | prepared_station_input | 2 | true | - Focus station relation rejected 'DSC_5333.NEF'<->'DSC_5434.NEF': scale 1.257, overlap 36.0%, NCC 0.938, edges 0.650/0.909, samples 1588；- Focus station relation rejected 'DSC_5333.NEF'<->'DSC_5422.NEF': scale 1.172, overlap 36.0%, NCC 0.997, edges 0.751/0.794, samples 2160；- Focus station relation rejected 'DSC_5333.NEF'<->'DSC_5423.NEF': scale 1.167, overlap 36.0%, NCC 0.996, edges 0.762/0.854, samples 2160；- Focus station relation rejected 'DSC_5333.NEF'<->'DSC_5431.NEF': scale 1.298, overlap 36.0%, NCC 0.879, edges 0.724/0.824, samples 1497；- Focus station relation rejected 'DSC_5333.NEF'<->'DSC_5425.NEF': scale 1.258, overlap 24.0%, NCC 0.992, edges 0.616/0.723, samples 1300；- Focus station relation rejected 'DSC_5333.NEF'<->'DSC_5439.NEF': scale 1.443, overlap 8.0%, NCC 0.905, edges 0.554/0.913, samples 275；- Focus station relation rejected 'DSC_5333.NEF'<->'DSC_5428.NEF': scale 1.328, overlap 12.0%, NCC 0.959, edges 0.290/0.657, samples 447；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5333.NEF': Local, 2974 inliers of 4303 control points, coverage 0.622, rejected 0.309, median symmetric error 0.23px, control point spacing 23.7px |
| DSC_5334.NEF | prepared_station_input | 4 | true | - Focus station relation rejected 'DSC_5420.NEF'<->'DSC_5334.NEF': scale 0.857, overlap 36.0%, NCC 0.822, edges 0.583/0.711, samples 1863；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5334.NEF': Local, 3145 inliers of 4329 control points, coverage 0.656, rejected 0.274, median symmetric error 0.23px, control point spacing 23.8px |
| DSC_5335.NEF | prepared_station_input | 10 | true | - Focus station relation rejected 'DSC_5335.NEF'<->'DSC_5421.NEF': scale 1.088, overlap 48.0%, NCC 0.985, edges 0.651/0.850, samples 1320；- Focus station relation rejected 'DSC_5335.NEF'<->'DSC_5423.NEF': scale 1.081, overlap 36.0%, NCC 0.986, edges 0.617/0.827, samples 1380；- Focus station relation rejected 'DSC_5335.NEF'<->'DSC_5432.NEF': scale 1.193, overlap 48.0%, NCC 0.867, edges 0.633/0.893, samples 1812；- Focus station relation rejected 'DSC_5335.NEF'<->'DSC_5436.NEF': scale 1.259, overlap 24.0%, NCC 0.870, edges 0.683/0.882, samples 1324；- Focus station relation rejected 'DSC_5335.NEF'<->'DSC_5424.NEF': scale 1.177, overlap 36.0%, NCC 0.996, edges 0.697/0.745, samples 2160；- Focus station relation rejected 'DSC_5335.NEF'<->'DSC_5428.NEF': scale 1.230, overlap 36.0%, NCC 0.994, edges 0.429/0.694, samples 1620 |
| DSC_5336.NEF | prepared_station_input | 10 | true | - Focus station relation rejected 'DSC_5336.NEF'<->'DSC_5433.NEF': scale 1.156, overlap 24.0%, NCC 0.788, edges 0.510/0.923, samples 660；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5336.NEF': Local, 3677 inliers of 4381 control points, coverage 0.767, rejected 0.161, median symmetric error 0.31px, control point spacing 23.7px |
| DSC_5337.NEF | prepared_station_input | 10 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5337.NEF': Local, 3464 inliers of 4356 control points, coverage 0.724, rejected 0.205, median symmetric error 0.34px, control point spacing 23.7px |
| DSC_5338.NEF | prepared_station_input | 10 | true | - Focus station relation rejected 'DSC_5338.NEF'<->'DSC_5429.NEF': scale 1.254, overlap 36.0%, NCC 0.829, edges 0.614/0.837, samples 1402；- Focus station relation rejected 'DSC_5338.NEF'<->'DSC_5438.NEF': scale 1.318, overlap 24.0%, NCC 0.826, edges 0.712/0.840, samples 938；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5338.NEF': Local, 3728 inliers of 4363 control points, coverage 0.777, rejected 0.146, median symmetric error 0.28px, control point spacing 23.7px |
| DSC_5340.NEF | prepared_station_input | 12 | true | - Focus station relation rejected 'DSC_5340.NEF'<->'DSC_5420.NEF': scale 1.037, overlap 32.0%, NCC 0.957, edges 0.388/0.839, samples 524；- Focus station relation rejected 'DSC_5340.NEF'<->'DSC_5426.NEF': scale 1.118, overlap 36.0%, NCC 0.978, edges 0.492/0.579, samples 1692；- Focus station relation rejected 'DSC_5340.NEF'<->'DSC_5438.NEF': scale 1.272, overlap 24.0%, NCC 0.887, edges 0.456/0.764, samples 1246；- Focus station relation rejected 'DSC_5340.NEF'<->'DSC_5428.NEF': scale 1.175, overlap 36.0%, NCC 0.995, edges 0.651/0.773, samples 2160 |
| DSC_5341.NEF | prepared_station_input | 12 | true | - Focus station relation rejected 'DSC_5341.NEF'<->'DSC_5435.NEF': scale 1.112, overlap 12.0%, NCC 0.805, edges 0.611/0.876, samples 164；- Focus station relation rejected 'DSC_5341.NEF'<->'DSC_5421.NEF': scale 1.034, overlap 24.0%, NCC 0.940, edges 0.474/0.907, samples 523；- Focus station relation rejected 'DSC_5341.NEF'<->'DSC_5432.NEF': scale 1.134, overlap 44.0%, NCC 0.828, edges 0.609/0.729, samples 1083；- Focus station relation rejected 'DSC_5341.NEF'<->'DSC_5436.NEF': scale 1.198, overlap 36.0%, NCC 0.823, edges 0.617/0.806, samples 1071；- Focus station relation rejected 'DSC_5341.NEF'<->'DSC_5430.NEF': scale 1.193, overlap 36.0%, NCC 0.926, edges 0.664/0.718, samples 1899；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5341.NEF': Local, 3743 inliers of 4282 control points, coverage 0.794, rejected 0.126, median symmetric error 0.28px, control point spacing 23.6px |
| DSC_5342.NEF | prepared_station_input | 14 | true | - Focus station relation rejected 'DSC_5434.NEF'<->'DSC_5342.NEF': scale 0.919, overlap 16.0%, NCC 0.429, edges 0.208/0.526, samples 2135；- Focus station relation rejected 'DSC_5426.NEF'<->'DSC_5342.NEF': scale 0.919, overlap 36.0%, NCC 0.647, edges 0.376/0.698, samples 724；- Focus station relation rejected 'DSC_5429.NEF'<->'DSC_5342.NEF': scale 0.847, overlap 36.0%, NCC 0.896, edges 0.490/0.627, samples 1530；- Focus station relation rejected 'DSC_5342.NEF'<->'DSC_5440.NEF': scale 1.247, overlap 36.0%, NCC 0.958, edges 0.347/0.413, samples 2160 |
| DSC_5343.NEF | prepared_station_input | 14 | true | - Focus station relation rejected 'DSC_5422.NEF'<->'DSC_5343.NEF': scale 0.991, overlap 12.0%, NCC 0.695, edges 0.656/0.924, samples 245；- Focus station relation rejected 'DSC_5423.NEF'<->'DSC_5343.NEF': scale 0.997, overlap 12.0%, NCC 0.437, edges 0.497/0.884, samples 280；- Focus station relation rejected 'DSC_5431.NEF'<->'DSC_5343.NEF': scale 0.886, overlap 36.0%, NCC 0.266, edges 0.524/0.729, samples 844；- Focus station relation rejected 'DSC_5437.NEF'<->'DSC_5343.NEF': scale 0.850, overlap 36.0%, NCC 0.835, edges 0.308/0.467, samples 1210；- Focus station relation rejected 'DSC_5427.NEF'<->'DSC_5343.NEF': scale 0.878, overlap 24.0%, NCC 0.785, edges 0.415/0.735, samples 868；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5343.NEF': Local, 3319 inliers of 4254 control points, coverage 0.714, rejected 0.220, median symmetric error 0.27px, control point spacing 23.6px |
| DSC_5344.NEF | prepared_station_input | 13 | true | - Focus station relation rejected 'DSC_5435.NEF'<->'DSC_5344.NEF': scale 0.917, overlap 16.0%, NCC 0.412, edges 0.220/0.698, samples 2118；- Focus station relation rejected 'DSC_5431.NEF'<->'DSC_5344.NEF': scale 0.888, overlap 36.0%, NCC 0.552, edges 0.534/0.688, samples 847；- Focus station relation rejected 'DSC_5437.NEF'<->'DSC_5344.NEF': scale 0.851, overlap 36.0%, NCC 0.733, edges 0.399/0.436, samples 1271；- Focus station relation rejected 'DSC_5424.NEF'<->'DSC_5344.NEF': scale 0.913, overlap 24.0%, NCC 0.658, edges 0.512/0.731, samples 700；- Focus station relation rejected 'DSC_5344.NEF'<->'DSC_5440.NEF': scale 1.244, overlap 36.0%, NCC 0.957, edges 0.349/0.468, samples 2160 |
| DSC_5345.NEF | prepared_station_input | 13 | true | - Focus station relation rejected 'DSC_5422.NEF'<->'DSC_5345.NEF': scale 0.997, overlap 12.0%, NCC 0.833, edges 0.740/0.748, samples 247；- Focus station relation rejected 'DSC_5420.NEF'<->'DSC_5345.NEF': scale 0.998, overlap 12.0%, NCC 0.746, edges 0.384/0.862, samples 245；- Focus station relation rejected 'DSC_5430.NEF'<->'DSC_5345.NEF': scale 0.860, overlap 36.0%, NCC 0.910, edges 0.727/0.610, samples 1488；- Focus station relation rejected 'DSC_5427.NEF'<->'DSC_5345.NEF': scale 0.883, overlap 24.0%, NCC 0.857, edges 0.725/0.732, samples 823；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5345.NEF': Local, 3301 inliers of 4326 control points, coverage 0.702, rejected 0.237, median symmetric error 0.27px, control point spacing 23.6px |
| DSC_5346.NEF | prepared_station_input | 8 | true | - Focus station relation rejected 'DSC_5423.NEF'<->'DSC_5346.NEF': scale 0.941, overlap 36.0%, NCC 0.861, edges 0.716/0.890, samples 766；- Focus station relation rejected 'DSC_5424.NEF'<->'DSC_5346.NEF': scale 0.859, overlap 32.0%, NCC 0.745, edges 0.570/0.844, samples 968 |
| DSC_5347.NEF | prepared_station_input | 8 | true | - Focus station relation rejected 'DSC_5347.NEF'<->'DSC_5433.NEF': scale 1.157, overlap 24.0%, NCC 0.913, edges 0.360/0.793, samples 1042；- Focus station relation rejected 'DSC_5347.NEF'<->'DSC_5432.NEF': scale 1.186, overlap 60.0%, NCC 0.782, edges 0.705/0.833, samples 1980；- Focus station relation rejected 'DSC_5347.NEF'<->'DSC_5429.NEF': scale 1.256, overlap 36.0%, NCC 0.706, edges 0.574/0.780, samples 1456；- Focus station relation rejected 'DSC_5347.NEF'<->'DSC_5428.NEF': scale 1.227, overlap 24.0%, NCC 0.582, edges 0.367/0.823, samples 906；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5347.NEF': Local, 3416 inliers of 4351 control points, coverage 0.726, rejected 0.215, median symmetric error 0.25px, control point spacing 23.6px |
| DSC_5348.NEF | prepared_station_input | 8 | true | - Focus station relation rejected 'DSC_5348.NEF'<->'DSC_5422.NEF': scale 1.083, overlap 36.0%, NCC 0.742, edges 0.825/0.861, samples 938；- Focus station relation rejected 'DSC_5348.NEF'<->'DSC_5436.NEF': scale 1.254, overlap 36.0%, NCC 0.958, edges 0.375/0.623, samples 2160；- Focus station relation rejected 'DSC_5348.NEF'<->'DSC_5439.NEF': scale 1.331, overlap 36.0%, NCC 0.953, edges 0.578/0.432, samples 1538；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5348.NEF': Local, 3387 inliers of 4300 control points, coverage 0.713, rejected 0.212, median symmetric error 0.23px, control point spacing 23.7px |
| DSC_5349.NEF | prepared_station_input | 8 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5349.NEF': Local, 3298 inliers of 4311 control points, coverage 0.697, rejected 0.235, median symmetric error 0.23px, control point spacing 23.7px |
| DSC_5350.NEF | prepared_station_input | 3 | true | - Focus station relation rejected 'DSC_5437.NEF'<->'DSC_5350.NEF': scale 0.745, overlap 32.0%, NCC 0.857, edges 0.380/0.612, samples 864；- Focus station relation rejected 'DSC_5426.NEF'<->'DSC_5350.NEF': scale 0.807, overlap 36.0%, NCC 0.916, edges 0.431/0.748, samples 596；- Focus station relation rejected 'DSC_5427.NEF'<->'DSC_5350.NEF': scale 0.770, overlap 8.0%, NCC 0.199, edges 0.738/0.679, samples 178 |
| DSC_5351.NEF | prepared_station_input | 3 | true | - Focus station relation rejected 'DSC_5351.NEF'<->'DSC_5433.NEF': scale 1.240, overlap 48.0%, NCC 0.972, edges 0.511/0.596, samples 2160；- Focus station relation rejected 'DSC_5351.NEF'<->'DSC_5422.NEF': scale 1.161, overlap 36.0%, NCC 0.923, edges 0.691/0.910, samples 1633；- Focus station relation rejected 'DSC_5351.NEF'<->'DSC_5420.NEF': scale 1.159, overlap 36.0%, NCC 0.927, edges 0.726/0.861, samples 1651；- Focus station relation rejected 'DSC_5351.NEF'<->'DSC_5431.NEF': scale 1.283, overlap 48.0%, NCC 0.827, edges 0.610/0.738, samples 1703；- Focus station relation rejected 'DSC_5351.NEF'<->'DSC_5429.NEF': scale 1.347, overlap 12.0%, NCC 0.968, edges 0.472/0.809, samples 480；- Focus station relation rejected 'DSC_5351.NEF'<->'DSC_5438.NEF': scale 1.413, overlap 12.0%, NCC 0.936, edges 0.377/0.602, samples 464；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5351.NEF': Local, 2974 inliers of 4310 control points, coverage 0.629, rejected 0.310, median symmetric error 0.23px, control point spacing 23.7px |
| DSC_5352.NEF | prepared_station_input | 1 | true | decoded source registered in Capture_Station; run stopped before output, no final pixel contribution claimed |
| DSC_5353.NEF | prepared_station_input | 1 | true | - Focus station relation rejected 'DSC_5353.NEF'<->'DSC_5433.NEF': scale 1.242, overlap 48.0%, NCC 0.973, edges 0.542/0.644, samples 2160；- Focus station relation rejected 'DSC_5353.NEF'<->'DSC_5422.NEF': scale 1.162, overlap 36.0%, NCC 0.923, edges 0.765/0.927, samples 1630；- Focus station relation rejected 'DSC_5353.NEF'<->'DSC_5420.NEF': scale 1.160, overlap 36.0%, NCC 0.927, edges 0.729/0.897, samples 1649；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5353.NEF': Local, 2858 inliers of 4342 control points, coverage 0.600, rejected 0.342, median symmetric error 0.23px, control point spacing 23.7px |
| DSC_5354.NEF | prepared_station_input | 1 | true | - Focus station relation rejected 'DSC_5354.NEF'<->'DSC_5432.NEF': scale 1.265, overlap 48.0%, NCC 0.830, edges 0.615/0.788, samples 1740；- Focus station relation rejected 'DSC_5354.NEF'<->'DSC_5430.NEF': scale 1.331, overlap 8.0%, NCC 0.953, edges 0.535/0.784, samples 480；- Focus station relation rejected 'DSC_5354.NEF'<->'DSC_5439.NEF': scale 1.414, overlap 12.0%, NCC 0.898, edges 0.520/0.584, samples 404；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5354.NEF': Local, 2861 inliers of 4408 control points, coverage 0.593, rejected 0.351, median symmetric error 0.23px, control point spacing 23.7px |
| DSC_5355.NEF | prepared_station_input | 1 | true | - Focus geometry retry candidate rejected edge 'DSC_5355.NEF'<->'DSC_5419.NEF': scale 1.166, overlap 36.0%, NCC 0.913, edges 0.428/0.824, samples 1629；- Focus station relation rejected 'DSC_5437.NEF'<->'DSC_5355.NEF': scale 0.750, overlap 24.0%, NCC 0.854, edges 0.530/0.604, samples 799；- Focus station relation rejected 'DSC_5425.NEF'<->'DSC_5355.NEF': scale 0.816, overlap 24.0%, NCC 0.868, edges 0.550/0.689, samples 530；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5355.NEF': Local, 2641 inliers of 4340 control points, coverage 0.549, rejected 0.391, median symmetric error 0.23px, control point spacing 23.6px |
| DSC_5419.NEF | prepared_station_input | 6 | true | - Focus geometry retry candidate rejected edge 'DSC_5355.NEF'<->'DSC_5419.NEF': scale 1.166, overlap 36.0%, NCC 0.913, edges 0.428/0.824, samples 1629 |
| DSC_5420.NEF | prepared_station_input | 6 | true | - Focus station relation rejected 'DSC_5353.NEF'<->'DSC_5420.NEF': scale 1.160, overlap 36.0%, NCC 0.927, edges 0.729/0.897, samples 1649；- Focus station relation rejected 'DSC_5351.NEF'<->'DSC_5420.NEF': scale 1.159, overlap 36.0%, NCC 0.927, edges 0.726/0.861, samples 1651；- Focus station relation rejected 'DSC_5420.NEF'<->'DSC_5334.NEF': scale 0.857, overlap 36.0%, NCC 0.822, edges 0.583/0.711, samples 1863；- Focus station relation rejected 'DSC_5340.NEF'<->'DSC_5420.NEF': scale 1.037, overlap 32.0%, NCC 0.957, edges 0.388/0.839, samples 524；- Focus station relation rejected 'DSC_5420.NEF'<->'DSC_5345.NEF': scale 0.998, overlap 12.0%, NCC 0.746, edges 0.384/0.862, samples 245；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5420.NEF': Local, 3530 inliers of 4236 control points, coverage 0.751, rejected 0.167, median symmetric error 0.27px, control point spacing 23.8px |
| DSC_5421.NEF | prepared_station_input | 5 | true | - Focus station relation rejected 'DSC_5335.NEF'<->'DSC_5421.NEF': scale 1.088, overlap 48.0%, NCC 0.985, edges 0.651/0.850, samples 1320；- Focus station relation rejected 'DSC_5341.NEF'<->'DSC_5421.NEF': scale 1.034, overlap 24.0%, NCC 0.940, edges 0.474/0.907, samples 523 |
| DSC_5422.NEF | prepared_station_input | 5 | true | - Focus station relation rejected 'DSC_5353.NEF'<->'DSC_5422.NEF': scale 1.162, overlap 36.0%, NCC 0.923, edges 0.765/0.927, samples 1630；- Focus station relation rejected 'DSC_5333.NEF'<->'DSC_5422.NEF': scale 1.172, overlap 36.0%, NCC 0.997, edges 0.751/0.794, samples 2160；- Focus station relation rejected 'DSC_5351.NEF'<->'DSC_5422.NEF': scale 1.161, overlap 36.0%, NCC 0.923, edges 0.691/0.910, samples 1633；- Focus station relation rejected 'DSC_5422.NEF'<->'DSC_5332.NEF': scale 0.855, overlap 36.0%, NCC 0.868, edges 0.455/0.728, samples 1851；- Focus station relation rejected 'DSC_5348.NEF'<->'DSC_5422.NEF': scale 1.083, overlap 36.0%, NCC 0.742, edges 0.825/0.861, samples 938；- Focus station relation rejected 'DSC_5422.NEF'<->'DSC_5345.NEF': scale 0.997, overlap 12.0%, NCC 0.833, edges 0.740/0.748, samples 247；- Focus station relation rejected 'DSC_5422.NEF'<->'DSC_5343.NEF': scale 0.991, overlap 12.0%, NCC 0.695, edges 0.656/0.924, samples 245；- Focus station relation rejected 'DSC_5422.NEF'<->'DSC_5438.NEF': scale 1.225, overlap 8.0%, NCC 0.729, edges 0.442/0.745, samples 140；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5422.NEF': Local, 3177 inliers of 4299 control points, coverage 0.673, rejected 0.261, median symmetric error 0.31px, control point spacing 23.7px |
| DSC_5423.NEF | prepared_station_input | 6 | true | - Focus station relation rejected 'DSC_5333.NEF'<->'DSC_5423.NEF': scale 1.167, overlap 36.0%, NCC 0.996, edges 0.762/0.854, samples 2160；- Focus station relation rejected 'DSC_5423.NEF'<->'DSC_5346.NEF': scale 0.941, overlap 36.0%, NCC 0.861, edges 0.716/0.890, samples 766；- Focus station relation rejected 'DSC_5335.NEF'<->'DSC_5423.NEF': scale 1.081, overlap 36.0%, NCC 0.986, edges 0.617/0.827, samples 1380；- Focus station relation rejected 'DSC_5423.NEF'<->'DSC_5343.NEF': scale 0.997, overlap 12.0%, NCC 0.437, edges 0.497/0.884, samples 280；- Focus station relation rejected 'DSC_5423.NEF'<->'DSC_5429.NEF': scale 1.170, overlap 12.0%, NCC 0.402, edges 0.220/0.870, samples 311；- Focus station relation rejected 'DSC_5423.NEF'<->'DSC_5440.NEF': scale 1.243, overlap 8.0%, NCC 0.695, edges 0.672/0.702, samples 175；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5423.NEF': Local, 3168 inliers of 4148 control points, coverage 0.668, rejected 0.236, median symmetric error 0.27px, control point spacing 23.9px |
| DSC_5424.NEF | prepared_station_input | 11 | true | - Focus station relation rejected 'DSC_5424.NEF'<->'DSC_5346.NEF': scale 0.859, overlap 32.0%, NCC 0.745, edges 0.570/0.844, samples 968；- Focus station relation rejected 'DSC_5335.NEF'<->'DSC_5424.NEF': scale 1.177, overlap 36.0%, NCC 0.996, edges 0.697/0.745, samples 2160；- Focus station relation rejected 'DSC_5424.NEF'<->'DSC_5344.NEF': scale 0.913, overlap 24.0%, NCC 0.658, edges 0.512/0.731, samples 700 |
| DSC_5425.NEF | prepared_station_input | 11 | true | - Focus station relation rejected 'DSC_5425.NEF'<->'DSC_5355.NEF': scale 0.816, overlap 24.0%, NCC 0.868, edges 0.550/0.689, samples 530；- Focus station relation rejected 'DSC_5333.NEF'<->'DSC_5425.NEF': scale 1.258, overlap 24.0%, NCC 0.992, edges 0.616/0.723, samples 1300；- Focus station relation rejected 'DSC_5425.NEF'<->'DSC_5331.NEF': scale 0.793, overlap 24.0%, NCC 0.801, edges 0.511/0.696, samples 865；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5425.NEF': Local, 3939 inliers of 4286 control points, coverage 0.836, rejected 0.081, median symmetric error 0.31px, control point spacing 23.7px |
| DSC_5426.NEF | prepared_station_input | 11 | true | - Focus station relation rejected 'DSC_5426.NEF'<->'DSC_5350.NEF': scale 0.807, overlap 36.0%, NCC 0.916, edges 0.431/0.748, samples 596；- Focus station relation rejected 'DSC_5340.NEF'<->'DSC_5426.NEF': scale 1.118, overlap 36.0%, NCC 0.978, edges 0.492/0.579, samples 1692；- Focus station relation rejected 'DSC_5426.NEF'<->'DSC_5342.NEF': scale 0.919, overlap 36.0%, NCC 0.647, edges 0.376/0.698, samples 724；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5426.NEF': Local, 3356 inliers of 4298 control points, coverage 0.713, rejected 0.219, median symmetric error 0.36px, control point spacing 23.8px |
| DSC_5427.NEF | prepared_station_input | 17 | true | - Focus station relation rejected 'DSC_5427.NEF'<->'DSC_5350.NEF': scale 0.770, overlap 8.0%, NCC 0.199, edges 0.738/0.679, samples 178；- Focus station relation rejected 'DSC_5427.NEF'<->'DSC_5332.NEF': scale 0.755, overlap 12.0%, NCC 0.576, edges 0.389/0.589, samples 244；- Focus station relation rejected 'DSC_5427.NEF'<->'DSC_5345.NEF': scale 0.883, overlap 24.0%, NCC 0.857, edges 0.725/0.732, samples 823；- Focus station relation rejected 'DSC_5427.NEF'<->'DSC_5343.NEF': scale 0.878, overlap 24.0%, NCC 0.785, edges 0.415/0.735, samples 868 |
| DSC_5428.NEF | prepared_station_input | 17 | true | - Focus station relation rejected 'DSC_5333.NEF'<->'DSC_5428.NEF': scale 1.328, overlap 12.0%, NCC 0.959, edges 0.290/0.657, samples 447；- Focus station relation rejected 'DSC_5347.NEF'<->'DSC_5428.NEF': scale 1.227, overlap 24.0%, NCC 0.582, edges 0.367/0.823, samples 906；- Focus station relation rejected 'DSC_5335.NEF'<->'DSC_5428.NEF': scale 1.230, overlap 36.0%, NCC 0.994, edges 0.429/0.694, samples 1620；- Focus station relation rejected 'DSC_5340.NEF'<->'DSC_5428.NEF': scale 1.175, overlap 36.0%, NCC 0.995, edges 0.651/0.773, samples 2160；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5428.NEF': Local, 3937 inliers of 4347 control points, coverage 0.847, rejected 0.094, median symmetric error 0.31px, control point spacing 23.6px |
| DSC_5429.NEF | prepared_station_input | 15 | true | - Focus station relation rejected 'DSC_5330.NEF'<->'DSC_5429.NEF': scale 1.366, overlap 12.0%, NCC 0.864, edges 0.334/0.940, samples 460；- Focus station relation rejected 'DSC_5351.NEF'<->'DSC_5429.NEF': scale 1.347, overlap 12.0%, NCC 0.968, edges 0.472/0.809, samples 480；- Focus station relation rejected 'DSC_5331.NEF'<->'DSC_5429.NEF': scale 1.365, overlap 12.0%, NCC 0.718, edges 0.651/0.862, samples 470；- Focus station relation rejected 'DSC_5423.NEF'<->'DSC_5429.NEF': scale 1.170, overlap 12.0%, NCC 0.402, edges 0.220/0.870, samples 311；- Focus station relation rejected 'DSC_5347.NEF'<->'DSC_5429.NEF': scale 1.256, overlap 36.0%, NCC 0.706, edges 0.574/0.780, samples 1456；- Focus station relation rejected 'DSC_5338.NEF'<->'DSC_5429.NEF': scale 1.254, overlap 36.0%, NCC 0.829, edges 0.614/0.837, samples 1402；- Focus station relation rejected 'DSC_5429.NEF'<->'DSC_5342.NEF': scale 0.847, overlap 36.0%, NCC 0.896, edges 0.490/0.627, samples 1530 |
| DSC_5430.NEF | prepared_station_input | 15 | true | - Focus station relation rejected 'DSC_5354.NEF'<->'DSC_5430.NEF': scale 1.331, overlap 8.0%, NCC 0.953, edges 0.535/0.784, samples 480；- Focus station relation rejected 'DSC_5341.NEF'<->'DSC_5430.NEF': scale 1.193, overlap 36.0%, NCC 0.926, edges 0.664/0.718, samples 1899；- Focus station relation rejected 'DSC_5430.NEF'<->'DSC_5345.NEF': scale 0.860, overlap 36.0%, NCC 0.910, edges 0.727/0.610, samples 1488；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5430.NEF': Local, 4018 inliers of 4289 control points, coverage 0.853, rejected 0.063, median symmetric error 0.31px, control point spacing 23.7px |
| DSC_5431.NEF | prepared_station_input | 7 | true | - Focus station relation rejected 'DSC_5333.NEF'<->'DSC_5431.NEF': scale 1.298, overlap 36.0%, NCC 0.879, edges 0.724/0.824, samples 1497；- Focus station relation rejected 'DSC_5351.NEF'<->'DSC_5431.NEF': scale 1.283, overlap 48.0%, NCC 0.827, edges 0.610/0.738, samples 1703；- Focus station relation rejected 'DSC_5431.NEF'<->'DSC_5344.NEF': scale 0.888, overlap 36.0%, NCC 0.552, edges 0.534/0.688, samples 847；- Focus station relation rejected 'DSC_5431.NEF'<->'DSC_5343.NEF': scale 0.886, overlap 36.0%, NCC 0.266, edges 0.524/0.729, samples 844 |
| DSC_5432.NEF | prepared_station_input | 7 | true | - Focus geometry retry candidate rejected edge 'DSC_5432.NEF'<->'DSC_5433.NEF': scale 0.974, overlap 48.0%, NCC 0.190, edges 0.157/0.863, samples 949；- Focus station relation rejected 'DSC_5354.NEF'<->'DSC_5432.NEF': scale 1.265, overlap 48.0%, NCC 0.830, edges 0.615/0.788, samples 1740；- Focus station relation rejected 'DSC_5432.NEF'<->'DSC_5332.NEF': scale 0.778, overlap 36.0%, NCC 0.719, edges 0.667/0.854, samples 971；- Focus station relation rejected 'DSC_5347.NEF'<->'DSC_5432.NEF': scale 1.186, overlap 60.0%, NCC 0.782, edges 0.705/0.833, samples 1980；- Focus station relation rejected 'DSC_5335.NEF'<->'DSC_5432.NEF': scale 1.193, overlap 48.0%, NCC 0.867, edges 0.633/0.893, samples 1812；- Focus station relation rejected 'DSC_5341.NEF'<->'DSC_5432.NEF': scale 1.134, overlap 44.0%, NCC 0.828, edges 0.609/0.729, samples 1083；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5432.NEF': Local, 3972 inliers of 4240 control points, coverage 0.862, rejected 0.063, median symmetric error 0.27px, control point spacing 23.7px |
| DSC_5433.NEF | prepared_station_input | 0 | true | - Focus geometry retry candidate rejected edge 'DSC_5432.NEF'<->'DSC_5433.NEF': scale 0.974, overlap 48.0%, NCC 0.190, edges 0.157/0.863, samples 949；- Focus station relation rejected 'DSC_5353.NEF'<->'DSC_5433.NEF': scale 1.242, overlap 48.0%, NCC 0.973, edges 0.542/0.644, samples 2160；- Focus station relation rejected 'DSC_5351.NEF'<->'DSC_5433.NEF': scale 1.240, overlap 48.0%, NCC 0.972, edges 0.511/0.596, samples 2160；- Focus station relation rejected 'DSC_5347.NEF'<->'DSC_5433.NEF': scale 1.157, overlap 24.0%, NCC 0.913, edges 0.360/0.793, samples 1042；- Focus station relation rejected 'DSC_5336.NEF'<->'DSC_5433.NEF': scale 1.156, overlap 24.0%, NCC 0.788, edges 0.510/0.923, samples 660 |
| DSC_5434.NEF | prepared_station_input | 0 | true | - Focus station relation rejected 'DSC_5333.NEF'<->'DSC_5434.NEF': scale 1.257, overlap 36.0%, NCC 0.938, edges 0.650/0.909, samples 1588；- Focus station relation rejected 'DSC_5434.NEF'<->'DSC_5332.NEF': scale 0.800, overlap 40.0%, NCC 0.925, edges 0.762/0.876, samples 1116；- Focus station relation rejected 'DSC_5434.NEF'<->'DSC_5342.NEF': scale 0.919, overlap 16.0%, NCC 0.429, edges 0.208/0.526, samples 2135；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5434.NEF': Local, 2782 inliers of 4148 control points, coverage 0.605, rejected 0.329, median symmetric error 0.31px, control point spacing 23.7px |
| DSC_5435.NEF | prepared_station_input | 0 | true | - Focus station relation rejected 'DSC_5341.NEF'<->'DSC_5435.NEF': scale 1.112, overlap 12.0%, NCC 0.805, edges 0.611/0.876, samples 164；- Focus station relation rejected 'DSC_5435.NEF'<->'DSC_5344.NEF': scale 0.917, overlap 16.0%, NCC 0.412, edges 0.220/0.698, samples 2118；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5435.NEF': Local, 3156 inliers of 4334 control points, coverage 0.661, rejected 0.272, median symmetric error 0.27px, control point spacing 23.7px |
| DSC_5436.NEF | prepared_station_input | 9 | true | - Focus station relation rejected 'DSC_5436.NEF'<->'DSC_5330.NEF': scale 0.738, overlap 24.0%, NCC 0.834, edges 0.831/0.819, samples 519；- Focus station relation rejected 'DSC_5436.NEF'<->'DSC_5332.NEF': scale 0.742, overlap 16.0%, NCC 0.768, edges 0.801/0.780, samples 522；- Focus station relation rejected 'DSC_5348.NEF'<->'DSC_5436.NEF': scale 1.254, overlap 36.0%, NCC 0.958, edges 0.375/0.623, samples 2160；- Focus station relation rejected 'DSC_5335.NEF'<->'DSC_5436.NEF': scale 1.259, overlap 24.0%, NCC 0.870, edges 0.683/0.882, samples 1324；- Focus station relation rejected 'DSC_5341.NEF'<->'DSC_5436.NEF': scale 1.198, overlap 36.0%, NCC 0.823, edges 0.617/0.806, samples 1071 |
| DSC_5437.NEF | prepared_station_input | 9 | true | - Focus station relation rejected 'DSC_5437.NEF'<->'DSC_5355.NEF': scale 0.750, overlap 24.0%, NCC 0.854, edges 0.530/0.604, samples 799；- Focus station relation rejected 'DSC_5437.NEF'<->'DSC_5350.NEF': scale 0.745, overlap 32.0%, NCC 0.857, edges 0.380/0.612, samples 864；- Focus station relation rejected 'DSC_5437.NEF'<->'DSC_5344.NEF': scale 0.851, overlap 36.0%, NCC 0.733, edges 0.399/0.436, samples 1271；- Focus station relation rejected 'DSC_5437.NEF'<->'DSC_5343.NEF': scale 0.850, overlap 36.0%, NCC 0.835, edges 0.308/0.467, samples 1210；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5437.NEF': Local, 3853 inliers of 4380 control points, coverage 0.805, rejected 0.120, median symmetric error 0.27px, control point spacing 23.7px |
| DSC_5438.NEF | prepared_station_input | 16 | true | - Focus station relation rejected 'DSC_5351.NEF'<->'DSC_5438.NEF': scale 1.413, overlap 12.0%, NCC 0.936, edges 0.377/0.602, samples 464；- Focus station relation rejected 'DSC_5422.NEF'<->'DSC_5438.NEF': scale 1.225, overlap 8.0%, NCC 0.729, edges 0.442/0.745, samples 140；- Focus station relation rejected 'DSC_5338.NEF'<->'DSC_5438.NEF': scale 1.318, overlap 24.0%, NCC 0.826, edges 0.712/0.840, samples 938；- Focus station relation rejected 'DSC_5340.NEF'<->'DSC_5438.NEF': scale 1.272, overlap 24.0%, NCC 0.887, edges 0.456/0.764, samples 1246 |
| DSC_5439.NEF | prepared_station_input | 16 | true | - Focus station relation rejected 'DSC_5354.NEF'<->'DSC_5439.NEF': scale 1.414, overlap 12.0%, NCC 0.898, edges 0.520/0.584, samples 404；- Focus station relation rejected 'DSC_5333.NEF'<->'DSC_5439.NEF': scale 1.443, overlap 8.0%, NCC 0.905, edges 0.554/0.913, samples 275；- Focus station relation rejected 'DSC_5348.NEF'<->'DSC_5439.NEF': scale 1.331, overlap 36.0%, NCC 0.953, edges 0.578/0.432, samples 1538；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5439.NEF': Local, 3988 inliers of 4366 control points, coverage 0.839, rejected 0.087, median symmetric error 0.23px, control point spacing 23.6px |
| DSC_5440.NEF | prepared_station_input | 16 | true | - Focus station relation rejected 'DSC_5331.NEF'<->'DSC_5440.NEF': scale 1.446, overlap 8.0%, NCC 0.892, edges 0.756/0.912, samples 319；- Focus station relation rejected 'DSC_5423.NEF'<->'DSC_5440.NEF': scale 1.243, overlap 8.0%, NCC 0.695, edges 0.672/0.702, samples 175；- Focus station relation rejected 'DSC_5344.NEF'<->'DSC_5440.NEF': scale 1.244, overlap 36.0%, NCC 0.957, edges 0.349/0.468, samples 2160；- Focus station relation rejected 'DSC_5342.NEF'<->'DSC_5440.NEF': scale 1.247, overlap 36.0%, NCC 0.958, edges 0.347/0.413, samples 2160；- Intra-station registration '/Users/dp/Downloads/书画拼图/马远2/DSC_5440.NEF': Local, 3914 inliers of 4382 control points, coverage 0.810, rejected 0.107, median symmetric error 0.27px, control point spacing 23.7px |

## 2. 实际路径、机位与停止阶段

- selected_path：`layered_virtual_tile`；station数：18；路径配置/回退：{"configured_path":null,"selected_path":"layered_virtual_tile","use_virtual_tiles":true,"fallback_reason":null}。
- isolated：[]；undecodable：[]；source_scan：{"singleton_station_count":0,"multi_frame_station_count":18,"singleton_ratio":0.0,"detected":false,"isolated_source_count":0}。
- closure："converged"；连通/分量：{"components":1,"component_members":[["/Users/dp/Downloads/书画拼图/马远2/DSC_5330.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5331.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5332.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5333.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5334.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5335.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5336.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5337.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5338.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5340.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5341.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5342.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5343.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5344.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5345.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5346.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5347.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5348.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5349.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5350.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5351.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5352.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5353.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5354.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5355.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5419.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5420.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5421.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5422.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5423.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5424.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5425.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5426.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5427.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5428.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5429.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5430.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5431.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5432.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5433.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5434.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5435.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5436.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5437.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5438.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5439.NEF","/Users/dp/Downloads/书画拼图/马远2/DSC_5440.NEF"]],"component_member_counts":[47]}。
- 输出密度/输出像素门：{"status":"rejected","factor":1.1767578125,"pixel_count_multiplier":1.3847589492797852,"canvas_pixels":0,"output_plane_lower_bound_bytes":0,"target_minimum_scale":1.0,"probe_step_px":32,"sampling_contract":"sampled_native_jacobian_not_full_pixel_proof","source_count":47,"sample_count":124543,"invalid_probe_count":0,"boundary_probe_count":3281,"worst_station_index":17,"worst_source_id":33,"worst_station_point":[2049.0,33.0],"direct_pair_p95_output_px":3.0650559059906275,"residual_median_output_px":0.5253280743027332,"max_corner_correction_output_px":5.466575463579199,"minimum_scale":0.8504955088348083,"p01_scale":0.8694860280443775,"median_scale":1.05552753496831,"bypass_reason":"output_pixel_closure_limit_exceeded"}。
- 结束：exit=1，analysis=`rejected`；最后完成/首次失败阶段：18机位内融合及机位关系求解完成，closure基础域converged；输出密度换算的output-pixel closure门拒绝，未进入最终合成/QG/预览导出。
- 原始失败：["harness failed (exit 101); see /Users/dp/Desktop/raw-editor/output/acceptance/task-5/mayuan-2-baseline-r1/run/mayuan-2-s0p25-task-5-baseline-r1.log","no focus-stack result line in the log","acceptance metrics were not written","full-resolution TIFF is missing or empty","the Stack_Report has no verified TIFF export at the requested path","verified TIFF dimensions do not match the rendered result","the TIFF export lacks verified unassociated coverage alpha (ExtraSamples=[2])","the image-stack dialog rejects this result (Stack_Report degradation.result is rejected)"]；degradation及native_errors完整保存在JSON，不用TIFF存在代替成功。
- QG：{"verdict":"not_run","roi_count":0,"criteria":{},"non_pass":[],"unresolved_pixels":null,"detail_repairs":0}。
- 独立tone状态：{"status":"not_run","boundary_delta_e":{"max":0.0,"threshold":1.5,"violations":[]},"correction_model":"station_affine","geometry_active_relation_count":0}。

## 3. N4.5失败链

以下数值直接来自本次报告。候选record、最终relation和hypothesis理由发生次数分母不同；同一pair可能有多次尝试，不直接相加。legacy路径空的station关系数组不代表注册无失败；未观测proposal/shared gate阶段标为unavailable。

```json
{
  "selected_path": "layered_virtual_tile",
  "candidates_recorded": 153,
  "model_fitted_records": 109,
  "accepted_relations": 66,
  "rejected_count": 87,
  "unmeasured_count": 44,
  "rejected_by_failure_stage": {
    "fitted": 43,
    "insufficient_bidirectional_matches": 9,
    "insufficient_grid_coverage": 1,
    "insufficient_measurable_patches": 34
  },
  "rejected_by_reason": {
    "closure_unreliable_pair_p95": 16,
    "station_relation_scale_out_of_range": 27
  },
  "candidate_stage_histogram": {
    "fitted": 109,
    "insufficient_measurable_patches": 34,
    "insufficient_bidirectional_matches": 9,
    "insufficient_grid_coverage": 1
  },
  "hypothesis_rejection_reason_occurrences": {
    "station_relation_scale_out_of_range": 246,
    "station_relation_low_spatial_support": 86,
    "station_relation_edge_strength_mismatch": 3,
    "station_relation_edge_orientation_mismatch": 32,
    "station_relation_high_reprojection_error": 1
  },
  "source_raw_pair_diagnostics_count": 0,
  "source_raw_direct_diagnostics_count": 0,
  "connectivity": {
    "components": 1,
    "component_members": [
      [
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5330.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5331.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5332.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5333.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5334.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5335.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5336.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5337.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5338.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5340.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5341.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5342.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5343.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5344.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5345.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5346.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5347.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5348.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5349.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5350.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5351.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5352.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5353.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5354.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5355.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5419.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5420.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5421.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5422.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5423.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5424.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5425.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5426.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5427.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5428.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5429.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5430.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5431.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5432.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5433.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5434.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5435.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5436.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5437.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5438.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5439.NEF",
        "/Users/dp/Downloads/书画拼图/马远2/DSC_5440.NEF"
      ]
    ],
    "component_member_counts": [
      47
    ]
  },
  "note": "Candidate records, final relation counts and hypothesis reason occurrences have different denominators and can repeat a pair. Empty legacy station diagnostics do not prove no registration failure. Shared-gate/proposal stages without explicit records are unavailable.",
  "proposal_count": null,
  "shared_gate_attempt_count": null,
  "station_chain_availability": "station candidate records only; source proposal/shared gate totals unavailable",
  "accepted_source_pairs_count": null,
  "alignment_cache": {
    "path": "/Users/dp/Desktop/raw-editor/output/acceptance/task-5/mayuan-2-baseline-r1/source-alignment.json",
    "schema": 2,
    "match_records": 1003
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
      "hypothesis_rejection_reason_occurrences": {
        "station_relation_low_spatial_support": 86
      }
    },
    "bidirectional": {
      "rejected_by_failure_stage": {
        "insufficient_bidirectional_matches": 9
      },
      "rejected_by_reason": {},
      "candidate_stage_histogram": {
        "insufficient_bidirectional_matches": 9
      },
      "hypothesis_rejection_reason_occurrences": {}
    },
    "grid": {
      "rejected_by_failure_stage": {
        "insufficient_grid_coverage": 1
      },
      "rejected_by_reason": {},
      "candidate_stage_histogram": {
        "insufficient_grid_coverage": 1
      },
      "hypothesis_rejection_reason_occurrences": {}
    },
    "scale": {
      "rejected_by_failure_stage": {},
      "rejected_by_reason": {
        "station_relation_scale_out_of_range": 27
      },
      "candidate_stage_histogram": {},
      "hypothesis_rejection_reason_occurrences": {
        "station_relation_scale_out_of_range": 246
      }
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
      "hypothesis_rejection_reason_occurrences": {
        "station_relation_edge_orientation_mismatch": 32
      }
    },
    "closure": {
      "rejected_by_failure_stage": {},
      "rejected_by_reason": {
        "closure_unreliable_pair_p95": 16
      },
      "candidate_stage_histogram": {},
      "hypothesis_rejection_reason_occurrences": {}
    }
  }
}
```

| 诊断类别 | 本次原始字段与统计口径 |
| --- | --- |
| no overlap | {"rejected_by_failure_stage":{},"rejected_by_reason":{},"candidate_stage_histogram":{},"hypothesis_rejection_reason_occurrences":{}} |
| support | {"rejected_by_failure_stage":{},"rejected_by_reason":{},"candidate_stage_histogram":{},"hypothesis_rejection_reason_occurrences":{"station_relation_low_spatial_support":86}} |
| bidirectional | {"rejected_by_failure_stage":{"insufficient_bidirectional_matches":9},"rejected_by_reason":{},"candidate_stage_histogram":{"insufficient_bidirectional_matches":9},"hypothesis_rejection_reason_occurrences":{}} |
| grid | {"rejected_by_failure_stage":{"insufficient_grid_coverage":1},"rejected_by_reason":{},"candidate_stage_histogram":{"insufficient_grid_coverage":1},"hypothesis_rejection_reason_occurrences":{}} |
| scale | {"rejected_by_failure_stage":{},"rejected_by_reason":{"station_relation_scale_out_of_range":27},"candidate_stage_histogram":{},"hypothesis_rejection_reason_occurrences":{"station_relation_scale_out_of_range":246}} |
| hull | {"rejected_by_failure_stage":{},"rejected_by_reason":{},"candidate_stage_histogram":{},"hypothesis_rejection_reason_occurrences":{}} |
| orientation | {"rejected_by_failure_stage":{},"rejected_by_reason":{},"candidate_stage_histogram":{},"hypothesis_rejection_reason_occurrences":{"station_relation_edge_orientation_mismatch":32}} |
| closure | {"rejected_by_failure_stage":{},"rejected_by_reason":{"closure_unreliable_pair_p95":16},"candidate_stage_histogram":{},"hypothesis_rejection_reason_occurrences":{}}；closure.status="converged" |

表内空map表示“未记录该类别的相应事件”，不能据此证明没有该类问题；未经过或未观测的门级计数为unavailable。

## 4. 资源与实际输出

- 实际原生Rayon线程：8；预处理池：[6]。
- 峰值RSS：13684703232bytes（12.744873046875GiB）；24GiB门：true；方法、采样数、native getrusage与报告值在resource.memory中。
- 缓存/解码计数：{"raw_decode_count":94,"native_fusion_count":0,"source_cache":{"max_bytes":0,"peak_bytes":0,"hits":0},"native_cache":{"max_bytes":0,"peak_bytes":0,"hits":0}}；阶段wall：{"preflight_seconds":0.405647584,"prepare_seconds":60.145733084,"matching_seconds":41.595237,"ordering_seconds":2.286661458,"render_seconds":0.0,"total_seconds":560.618368791}。
- wall：561.4903879165649s；contended：false。本次独占，但不同规模/路径/scale不得据单次耗时推断性能改善。
- 磁盘由run.py在启动前检查空闲≥40GiB、lane<20GiB；精确启动余量未记录。输出/报告/日志/预览的路径和SHA均在机器诊断。
- output：{"written":false,"format":"","path":"","width":0,"height":0,"bit_depth":16,"alpha_preserved":false,"icc":"sRGB","tiff_extra_samples":null,"artifact_validation":"not_run","artifact_validation_error":null,"result_id":"","preview":{"path":"","max_roi_delta_e":0.0}}；TIFF SHA：`not_written`。
- 缩略预览：[]；六区裁片见crop_previews。均来自scale=0.25渲染，不称原生1:1裁片。

## 5. 最明显问题与初步归因

未生成预览，不能定位可见接缝或声称已目视通过。数值阻断为output_density.factor=1.1767578125后direct_pair_p95_output_px=3.0650559059906275>3；report记录worst_station_index=17、worst_source_id=33、worst_station_point=[2049,33]。该坐标是station域的密度探针位置，**不是已证实的最坏闭环pair或成片瑕疵坐标**。基础闭环max_direct_pair_p95_px=2.6046616163770975，换算后超门，不能用基础域converged覆盖拒绝。

- QG非pass原始测量/失败坐标见机器诊断failed_criteria，保留world/owner_path原字段；无坐标时不自行转成output坐标。
- 初步区域组：站点求解组（任务16）；先核查输出密度与闭环单位换算、关系误差和证据支持。
- 任务7（置换检验）：建议作为关系证据诊断的辅助实验；目前连通不等于无假边，单次数据不能断言固定阈值应改。
- 任务8（provider对照）：可用于复核被拒关系的尺度/支持/方向召回；当前首要阻断是输出像素闭环门，尚不能证明更换provider即可解决。
- 任务15（必要性能）：本次561.49秒结束，峰值12.745GiB且内存门通过；没有超门或无法结束证据，不据此启动。

## 6. 范围化结论与交付

- 当前等级：未达D1（几何门拒绝，无预览）；未达D2–D5。scale=0.25不能授予D2/D3；无用户D4确认，也不授予D5。
- 输入/参考保持：见analysis checks；47张均解码并登记到18机位，输入与参考/二进制检查通过；无最终合成，47张仅记prepared_station_input，不虚构成片参与像素，也没有证据支持任意排除一张。
- 本卡仅覆盖本次全量清单与冻结二进制；不挪用10张子集通过结论，也不外推scale=1、其他线程数或并发场景。
- 复看：诊断结果未提交用户画质确认，队列维持0/3；预览不是已验收成片。
- L0结论：接受本轮摸底诊断与上方区域归组；修复按既有批次/依赖排队，保持全部拒绝及观测缺口，不授予原生或用户画质验收。
- 下一步：任务16归站点求解组；保留本次失败版本，先修输出密度/闭环阻断后重跑同范围，不放宽3px门限。
