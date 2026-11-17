# S2-sleeping：睡觉全量摸底

- 更新：2026-11-18；版本：baseline-r1；对应任务5；执行者/L0为本次同一执行者。
- 目的：只跑原冻结基线、记录失败和能力缺口；无算法、阈值或像素预期改动。
- 起点：v1 `1da9fd44`；工作区`/Users/dp/Desktop/raw-editor/.worktrees/task-5`；分支`lane/task-5/baseline`。
- 输入：已登记全量45张，scale=0.25；manifest为`tests/acceptance/focus-stack-datasets.json`，SHA `46b4f77fa6a41852223954aa9f6fe8eb92eee3b737b465fac1ff4e05d07e6173`；选中输入清单SHA `f3ab1237ccedf7c23c5e34bc0d53c43bcac210de33c070538304859ebe9681df`。
- 冻结：`base-2026-11-12`；原始提交`c0c1e5a9037fb4f8a8f8b983802848feb667b9ba`；二进制SHA `39bfe6f2aecc8a7ebc154efb94ba4b1f1a26257c825a676a0e54e8cde9cef03b`。build.json位于`/Users/dp/Desktop/raw-editor/output/acceptance/binaries/base-2026-11-12/build.json`。
- 资源：exclusive，Rayon≤8、jobs≤6、nice≥10；该版本未验证共享线程合同，兼容规则强制独占。未知全量峰值同样须独占。
- 原始输出：`/Users/dp/Desktop/raw-editor/output/acceptance/task-5/sleeping-baseline-r1`；机器诊断：[完整字段、全源清单与摘要](../benchmarks/sleeping-baseline-r1-diagnostic.json)。
- runner、工具、环境与原始命令绑定在execution.json；输入和参考只读，前后核对见analysis.json。

## 1. 来源参与与排除

- 分类计数：{"prepared_station_input":45}，互斥合计45/45；`detail_only`已去除同时streamed的帧。
- `composited`来自Streaming日志；`singleton_tile_input`是单帧机位直接来源，未执行多帧fusion；`station_fusion_input`来自虚拟tile provenance及已完成机位fusion，只证明参与机位融合，不宣称每帧持有最终像素；`detail_only`由最终detail_repairs的donor记录证明；仅Lending但无转移记录的帧单列为detail_candidate_only。
- `prepared_station_input`只证明解码/分组参与，运行未出图时不声明最终像素归属。`excluded`须逐项结合native_reason_lines解释；`unresolved`不能补成参与或排除。全部源图的绝对路径、输入SHA、排序/机位、解码及原生日志证据保存在机器诊断的sources数组。

| 源图文件名（完整路径见JSON） | 分类 | 机位 | 排序中 | 原始原因 |
| --- | --- | --- | --- | --- |
| DSC_6461.NEF | prepared_station_input | 3 | true | decoded source registered in Capture_Station; run stopped before output, no final pixel contribution claimed |
| DSC_6462.NEF | prepared_station_input | 3 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6462.NEF': Local, 3946 inliers of 4536 control points, coverage 0.814, rejected 0.130, median symmetric error 0.27px, control point spacing 23.8px |
| DSC_6463.NEF | prepared_station_input | 4 | true | decoded source registered in Capture_Station; run stopped before output, no final pixel contribution claimed |
| DSC_6464.NEF | prepared_station_input | 4 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6464.NEF': Local, 3864 inliers of 4565 control points, coverage 0.812, rejected 0.154, median symmetric error 0.31px, control point spacing 23.7px |
| DSC_6465.NEF | prepared_station_input | 4 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6465.NEF': Local, 3986 inliers of 4565 control points, coverage 0.837, rejected 0.127, median symmetric error 0.20px, control point spacing 23.7px |
| DSC_6466.NEF | prepared_station_input | 4 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6466.NEF': Local, 4023 inliers of 4620 control points, coverage 0.845, rejected 0.129, median symmetric error 0.20px, control point spacing 23.6px |
| DSC_6467.NEF | prepared_station_input | 6 | true | - Focus station relation rejected 'DSC_6467.NEF'<->'DSC_6487.NEF': scale 1.267, overlap 8.0%, NCC 0.754, edges 0.140/0.341, samples 1702 |
| DSC_6468.NEF | prepared_station_input | 6 | true | - Focus station relation rejected 'DSC_6468.NEF'<->'DSC_6483.NEF': scale 1.269, overlap 8.0%, NCC 0.797, edges 0.223/0.686, samples 1703；- Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6468.NEF': Local, 4038 inliers of 4497 control points, coverage 0.833, rejected 0.102, median symmetric error 0.23px, control point spacing 23.8px |
| DSC_6469.NEF | prepared_station_input | 8 | true | decoded source registered in Capture_Station; run stopped before output, no final pixel contribution claimed |
| DSC_6470.NEF | prepared_station_input | 9 | true | decoded source registered in Capture_Station; run stopped before output, no final pixel contribution claimed |
| DSC_6471.NEF | prepared_station_input | 9 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6471.NEF': Local, 4247 inliers of 4482 control points, coverage 0.892, rejected 0.052, median symmetric error 0.27px, control point spacing 23.7px |
| DSC_6472.NEF | prepared_station_input | 8 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6472.NEF': Local, 4365 inliers of 4565 control points, coverage 0.901, rejected 0.044, median symmetric error 0.23px, control point spacing 23.7px |
| DSC_6473.NEF | prepared_station_input | 9 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6473.NEF': Local, 4322 inliers of 4565 control points, coverage 0.911, rejected 0.053, median symmetric error 0.27px, control point spacing 23.7px |
| DSC_6474.NEF | prepared_station_input | 11 | true | - Focus station relation rejected 'DSC_6474.NEF'<->'DSC_6495.NEF': scale 0.854, overlap 8.0%, NCC 0.923, edges 0.375/0.550, samples 2493 |
| DSC_6475.NEF | prepared_station_input | 11 | true | - Focus station relation rejected 'DSC_6505.NEF'<->'DSC_6475.NEF': scale 1.284, overlap 8.0%, NCC 0.750, edges 0.157/0.740, samples 2235；- Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6475.NEF': Local, 3961 inliers of 4434 control points, coverage 0.825, rejected 0.107, median symmetric error 0.14px, control point spacing 23.8px |
| DSC_6476.NEF | prepared_station_input | 11 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6476.NEF': Local, 4006 inliers of 4513 control points, coverage 0.827, rejected 0.112, median symmetric error 0.18px, control point spacing 23.8px |
| DSC_6477.NEF | prepared_station_input | 11 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6477.NEF': Local, 4067 inliers of 4535 control points, coverage 0.839, rejected 0.103, median symmetric error 0.23px, control point spacing 23.7px |
| DSC_6478.NEF | prepared_station_input | 11 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6478.NEF': Local, 4044 inliers of 4423 control points, coverage 0.855, rejected 0.086, median symmetric error 0.23px, control point spacing 23.7px |
| DSC_6479.NEF | prepared_station_input | 14 | true | decoded source registered in Capture_Station; run stopped before output, no final pixel contribution claimed |
| DSC_6480.NEF | prepared_station_input | 14 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6480.NEF': Local, 4076 inliers of 4317 control points, coverage 0.888, rejected 0.056, median symmetric error 0.23px, control point spacing 23.9px |
| DSC_6481.NEF | prepared_station_input | 14 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6481.NEF': Local, 4117 inliers of 4548 control points, coverage 0.850, rejected 0.095, median symmetric error 0.23px, control point spacing 23.9px |
| DSC_6482.NEF | prepared_station_input | 14 | true | - Focus station relation rejected 'DSC_6489.NEF'<->'DSC_6482.NEF': scale 1.218, overlap 24.0%, NCC 0.957, edges 0.423/0.818, samples 483；- Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6482.NEF': Local, 3844 inliers of 4455 control points, coverage 0.803, rejected 0.137, median symmetric error 0.26px, control point spacing 23.8px |
| DSC_6483.NEF | prepared_station_input | 12 | true | - Focus station relation rejected 'DSC_6468.NEF'<->'DSC_6483.NEF': scale 1.269, overlap 8.0%, NCC 0.797, edges 0.223/0.686, samples 1703 |
| DSC_6484.NEF | prepared_station_input | 12 | true | - Focus station relation rejected 'DSC_6497.NEF'<->'DSC_6484.NEF': scale 1.241, overlap 12.0%, NCC 0.937, edges 0.445/0.897, samples 277；- Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6484.NEF': Local, 4151 inliers of 4541 control points, coverage 0.857, rejected 0.086, median symmetric error 0.32px, control point spacing 23.8px |
| DSC_6485.NEF | prepared_station_input | 12 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6485.NEF': Local, 4228 inliers of 4565 control points, coverage 0.888, rejected 0.074, median symmetric error 0.23px, control point spacing 23.7px |
| DSC_6486.NEF | prepared_station_input | 13 | true | - Focus station relation rejected 'DSC_6497.NEF'<->'DSC_6486.NEF': scale 1.240, overlap 12.0%, NCC 0.914, edges 0.285/0.650, samples 207 |
| DSC_6487.NEF | prepared_station_input | 13 | true | - Focus station relation rejected 'DSC_6467.NEF'<->'DSC_6487.NEF': scale 1.267, overlap 8.0%, NCC 0.754, edges 0.140/0.341, samples 1702；- Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6487.NEF': Local, 3774 inliers of 4482 control points, coverage 0.809, rejected 0.158, median symmetric error 0.23px, control point spacing 23.8px |
| DSC_6488.NEF | prepared_station_input | 10 | true | decoded source registered in Capture_Station; run stopped before output, no final pixel contribution claimed |
| DSC_6489.NEF | prepared_station_input | 10 | true | - Focus station relation rejected 'DSC_6489.NEF'<->'DSC_6482.NEF': scale 1.218, overlap 24.0%, NCC 0.957, edges 0.423/0.818, samples 483；- Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6489.NEF': Local, 4137 inliers of 4496 control points, coverage 0.854, rejected 0.080, median symmetric error 0.46px, control point spacing 23.6px |
| DSC_6490.NEF | prepared_station_input | 10 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6490.NEF': Local, 4351 inliers of 4620 control points, coverage 0.898, rejected 0.058, median symmetric error 0.28px, control point spacing 23.7px |
| DSC_6491.NEF | prepared_station_input | 10 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6491.NEF': Local, 4422 inliers of 4620 control points, coverage 0.929, rejected 0.043, median symmetric error 0.28px, control point spacing 23.7px |
| DSC_6492.NEF | prepared_station_input | 10 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6492.NEF': Local, 4366 inliers of 4536 control points, coverage 0.917, rejected 0.037, median symmetric error 0.28px, control point spacing 23.6px |
| DSC_6493.NEF | prepared_station_input | 5 | true | decoded source registered in Capture_Station; run stopped before output, no final pixel contribution claimed |
| DSC_6494.NEF | prepared_station_input | 5 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6494.NEF': Local, 4242 inliers of 4565 control points, coverage 0.891, rejected 0.071, median symmetric error 0.28px, control point spacing 23.7px |
| DSC_6495.NEF | prepared_station_input | 7 | true | - Focus station relation rejected 'DSC_6474.NEF'<->'DSC_6495.NEF': scale 0.854, overlap 8.0%, NCC 0.923, edges 0.375/0.550, samples 2493 |
| DSC_6496.NEF | prepared_station_input | 7 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6496.NEF': Local, 3854 inliers of 4399 control points, coverage 0.810, rejected 0.124, median symmetric error 0.40px, control point spacing 23.8px |
| DSC_6497.NEF | prepared_station_input | 7 | true | - Focus station relation rejected 'DSC_6497.NEF'<->'DSC_6484.NEF': scale 1.241, overlap 12.0%, NCC 0.937, edges 0.445/0.897, samples 277；- Focus station relation rejected 'DSC_6497.NEF'<->'DSC_6486.NEF': scale 1.240, overlap 12.0%, NCC 0.914, edges 0.285/0.650, samples 207；- Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6497.NEF': Local, 4172 inliers of 4552 control points, coverage 0.876, rejected 0.083, median symmetric error 0.23px, control point spacing 23.7px |
| DSC_6498.NEF | prepared_station_input | 1 | true | decoded source registered in Capture_Station; run stopped before output, no final pixel contribution claimed |
| DSC_6499.NEF | prepared_station_input | 1 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6499.NEF': Local, 3976 inliers of 4428 control points, coverage 0.845, rejected 0.102, median symmetric error 0.28px, control point spacing 23.8px |
| DSC_6500.NEF | prepared_station_input | 1 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6500.NEF': Local, 4194 inliers of 4620 control points, coverage 0.881, rejected 0.092, median symmetric error 0.27px, control point spacing 23.7px |
| DSC_6501.NEF | prepared_station_input | 0 | true | decoded source registered in Capture_Station; run stopped before output, no final pixel contribution claimed |
| DSC_6502.NEF | prepared_station_input | 0 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6502.NEF': Local, 3826 inliers of 4419 control points, coverage 0.790, rejected 0.134, median symmetric error 0.26px, control point spacing 23.9px |
| DSC_6503.NEF | prepared_station_input | 0 | true | - Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6503.NEF': Local, 3883 inliers of 4476 control points, coverage 0.801, rejected 0.132, median symmetric error 0.23px, control point spacing 23.9px |
| DSC_6504.NEF | prepared_station_input | 2 | true | decoded source registered in Capture_Station; run stopped before output, no final pixel contribution claimed |
| DSC_6505.NEF | prepared_station_input | 2 | true | - Focus station relation rejected 'DSC_6505.NEF'<->'DSC_6475.NEF': scale 1.284, overlap 8.0%, NCC 0.750, edges 0.157/0.740, samples 2235；- Intra-station registration '/Users/dp/Downloads/书画拼图/睡觉/DSC_6505.NEF': Local, 3596 inliers of 4536 control points, coverage 0.742, rejected 0.207, median symmetric error 0.36px, control point spacing 23.7px |

## 2. 实际路径、机位与停止阶段

- selected_path：`layered_virtual_tile`；station数：15；路径配置/回退：{"configured_path":null,"selected_path":"layered_virtual_tile","use_virtual_tiles":true,"fallback_reason":null}。
- isolated：[]；undecodable：[]；source_scan：{"singleton_station_count":0,"multi_frame_station_count":15,"singleton_ratio":0.0,"detected":false,"isolated_source_count":0}。
- closure："converged"；连通/分量：{"components":1,"component_members":[["/Users/dp/Downloads/书画拼图/睡觉/DSC_6461.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6462.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6463.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6464.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6465.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6466.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6467.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6468.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6469.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6470.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6471.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6472.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6473.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6474.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6475.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6476.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6477.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6478.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6479.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6480.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6481.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6482.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6483.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6484.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6485.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6486.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6487.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6488.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6489.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6490.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6491.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6492.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6493.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6494.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6495.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6496.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6497.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6498.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6499.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6500.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6501.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6502.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6503.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6504.NEF","/Users/dp/Downloads/书画拼图/睡觉/DSC_6505.NEF"]],"component_member_counts":[45]}。
- 输出密度/输出像素门：{"status":"rejected","factor":1.37890625,"pixel_count_multiplier":1.9013824462890625,"canvas_pixels":0,"output_plane_lower_bound_bytes":0,"target_minimum_scale":1.0,"probe_step_px":32,"sampling_contract":"sampled_native_jacobian_not_full_pixel_proof","source_count":45,"sample_count":124386,"invalid_probe_count":0,"boundary_probe_count":3318,"worst_station_index":14,"worst_source_id":21,"worst_station_point":[129.0,1345.0],"direct_pair_p95_output_px":3.500500400619984,"residual_median_output_px":0.41463193061341885,"max_corner_correction_output_px":5.112098209613919,"minimum_scale":0.7253261115764642,"p01_scale":0.7322722436856487,"median_scale":0.9028795124886823,"bypass_reason":"output_pixel_closure_limit_exceeded"}。
- 结束：exit=1，analysis=`rejected`；最后完成/首次失败阶段：15机位内融合及关系求解完成，基础域closure=converged；output_density的输出像素闭环门拒绝，未进入最终合成/QG/预览导出。
- 原始失败：["harness failed (exit 101); see /Users/dp/Desktop/raw-editor/output/acceptance/task-5/sleeping-baseline-r1/run/sleeping-s0p25-task-5-baseline-r1.log","no focus-stack result line in the log","acceptance metrics were not written","full-resolution TIFF is missing or empty","the Stack_Report has no verified TIFF export at the requested path","verified TIFF dimensions do not match the rendered result","the TIFF export lacks verified unassociated coverage alpha (ExtraSamples=[2])","the image-stack dialog rejects this result (Stack_Report degradation.result is rejected)"]；degradation及native_errors完整保存在JSON，不用TIFF存在代替成功。
- QG：{"verdict":"not_run","roi_count":0,"criteria":{},"non_pass":[],"unresolved_pixels":null,"detail_repairs":0}。
- 独立tone状态：{"status":"not_run","boundary_delta_e":{"max":0.0,"threshold":1.5,"violations":[]},"correction_model":"station_affine","geometry_active_relation_count":0}。

## 3. N4.5失败链

以下数值直接来自本次报告。候选record、最终relation和hypothesis理由发生次数分母不同；同一pair可能有多次尝试，不直接相加。legacy路径空的station关系数组不代表注册无失败；未观测proposal/shared gate阶段标为unavailable。

```json
{
  "selected_path": "layered_virtual_tile",
  "candidates_recorded": 105,
  "model_fitted_records": 73,
  "accepted_relations": 66,
  "rejected_count": 39,
  "unmeasured_count": 32,
  "rejected_by_failure_stage": {
    "fitted": 6,
    "insufficient_matches_after_model_quota": 1,
    "insufficient_measurable_patches": 30,
    "no_predicted_overlap": 2
  },
  "rejected_by_reason": {
    "closure_unreliable_pair_p95": 4,
    "station_relation_scale_out_of_range": 2
  },
  "candidate_stage_histogram": {
    "fitted": 72,
    "insufficient_matches_after_model_quota": 1,
    "insufficient_measurable_patches": 30,
    "no_predicted_overlap": 2
  },
  "hypothesis_rejection_reason_occurrences": {
    "station_relation_low_spatial_support": 59,
    "station_relation_scale_out_of_range": 77,
    "station_relation_edge_orientation_mismatch": 8,
    "station_relation_high_reprojection_error": 4
  },
  "source_raw_pair_diagnostics_count": 0,
  "source_raw_direct_diagnostics_count": 0,
  "connectivity": {
    "components": 1,
    "component_members": [
      [
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6461.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6462.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6463.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6464.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6465.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6466.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6467.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6468.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6469.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6470.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6471.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6472.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6473.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6474.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6475.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6476.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6477.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6478.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6479.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6480.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6481.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6482.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6483.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6484.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6485.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6486.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6487.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6488.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6489.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6490.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6491.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6492.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6493.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6494.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6495.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6496.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6497.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6498.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6499.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6500.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6501.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6502.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6503.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6504.NEF",
        "/Users/dp/Downloads/书画拼图/睡觉/DSC_6505.NEF"
      ]
    ],
    "component_member_counts": [
      45
    ]
  },
  "note": "Candidate records, final relation counts and hypothesis reason occurrences have different denominators and can repeat a pair. Empty legacy station diagnostics do not prove no registration failure. Shared-gate/proposal stages without explicit records are unavailable.",
  "proposal_count": null,
  "shared_gate_attempt_count": null,
  "station_chain_availability": "station candidate records only; source proposal/shared gate totals unavailable",
  "accepted_source_pairs_count": null,
  "alignment_cache": {
    "path": "/Users/dp/Desktop/raw-editor/output/acceptance/task-5/sleeping-baseline-r1/source-alignment.json",
    "schema": 2,
    "match_records": 671
  },
  "diagnostic_categories": {
    "no_overlap": {
      "rejected_by_failure_stage": {
        "no_predicted_overlap": 2
      },
      "rejected_by_reason": {},
      "candidate_stage_histogram": {
        "no_predicted_overlap": 2
      },
      "hypothesis_rejection_reason_occurrences": {}
    },
    "support": {
      "rejected_by_failure_stage": {},
      "rejected_by_reason": {},
      "candidate_stage_histogram": {},
      "hypothesis_rejection_reason_occurrences": {
        "station_relation_low_spatial_support": 59
      }
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
      "rejected_by_reason": {
        "station_relation_scale_out_of_range": 2
      },
      "candidate_stage_histogram": {},
      "hypothesis_rejection_reason_occurrences": {
        "station_relation_scale_out_of_range": 77
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
        "station_relation_edge_orientation_mismatch": 8
      }
    },
    "closure": {
      "rejected_by_failure_stage": {},
      "rejected_by_reason": {
        "closure_unreliable_pair_p95": 4
      },
      "candidate_stage_histogram": {},
      "hypothesis_rejection_reason_occurrences": {}
    }
  }
}
```

| 诊断类别 | 本次原始字段与统计口径 |
| --- | --- |
| no overlap | {"rejected_by_failure_stage":{"no_predicted_overlap":2},"rejected_by_reason":{},"candidate_stage_histogram":{"no_predicted_overlap":2},"hypothesis_rejection_reason_occurrences":{}} |
| support | {"rejected_by_failure_stage":{},"rejected_by_reason":{},"candidate_stage_histogram":{},"hypothesis_rejection_reason_occurrences":{"station_relation_low_spatial_support":59}} |
| bidirectional | {"rejected_by_failure_stage":{},"rejected_by_reason":{},"candidate_stage_histogram":{},"hypothesis_rejection_reason_occurrences":{}} |
| grid | {"rejected_by_failure_stage":{},"rejected_by_reason":{},"candidate_stage_histogram":{},"hypothesis_rejection_reason_occurrences":{}} |
| scale | {"rejected_by_failure_stage":{},"rejected_by_reason":{"station_relation_scale_out_of_range":2},"candidate_stage_histogram":{},"hypothesis_rejection_reason_occurrences":{"station_relation_scale_out_of_range":77}} |
| hull | {"rejected_by_failure_stage":{},"rejected_by_reason":{},"candidate_stage_histogram":{},"hypothesis_rejection_reason_occurrences":{}} |
| orientation | {"rejected_by_failure_stage":{},"rejected_by_reason":{},"candidate_stage_histogram":{},"hypothesis_rejection_reason_occurrences":{"station_relation_edge_orientation_mismatch":8}} |
| closure | {"rejected_by_failure_stage":{},"rejected_by_reason":{"closure_unreliable_pair_p95":4},"candidate_stage_histogram":{},"hypothesis_rejection_reason_occurrences":{}}；closure.status="converged" |

表内空map表示“未记录该类别的相应事件”，不能据此证明没有该类问题；未经过或未观测的门级计数为unavailable。

## 4. 资源与实际输出

- 实际原生Rayon线程：8；预处理池：[6]。
- 峰值RSS：13151518720bytes（12.248306274414062GiB）；24GiB门：true；方法、采样数、native getrusage与报告值在resource.memory中。
- 缓存/解码计数：{"raw_decode_count":90,"native_fusion_count":0,"source_cache":{"max_bytes":0,"peak_bytes":0,"hits":0},"native_cache":{"max_bytes":0,"peak_bytes":0,"hits":0}}；阶段wall：{"preflight_seconds":0.37232125,"prepare_seconds":56.807996708,"matching_seconds":28.75843225,"ordering_seconds":2.141266542,"render_seconds":0.0,"total_seconds":599.895768416}。
- wall：600.613107919693s；contended：false。本次独占，但不同规模/路径/scale不得据单次耗时推断性能改善。
- 磁盘由run.py在启动前检查空闲≥40GiB、lane<20GiB；精确启动余量未记录。输出/报告/日志/预览的路径和SHA均在机器诊断。
- output：{"written":false,"format":"","path":"","width":0,"height":0,"bit_depth":16,"alpha_preserved":false,"icc":"sRGB","tiff_extra_samples":null,"artifact_validation":"not_run","artifact_validation_error":null,"result_id":"","preview":{"path":"","max_roi_delta_e":0.0}}；TIFF SHA：`not_written`。
- 缩略预览：[]；六区裁片见crop_previews。均来自scale=0.25渲染，不称原生1:1裁片。

## 5. 最明显问题与初步归因

未生成预览，无法给出可见软区/接缝的成片坐标。数值阻断为factor=1.37890625后direct_pair_p95_output_px=3.500500400619984>3；基础域max_direct_pair_p95_px=2.5386065228292236。密度探针worst_station_index=14、worst_source_id=21、worst_station_point=[129,1345]仅是station域密度证据，不能当作已证实的最坏闭环pair或可见成片问题位置。

- QG非pass原始测量/失败坐标见机器诊断failed_criteria，保留world/owner_path原字段；无坐标时不自行转成output坐标。
- 初步区域组：站点求解组（任务16）；输出密度/坐标单位与闭环余量优先，与马远2的同类阻断一起调查。
- 任务7（置换检验）：作为假边/支持域对照的辅助实验；当前不能把单次输出门失败归咎于固定阈值。
- 任务8（provider对照）：可对照尺度/支持域/方向被拒关系；先确认输出像素门与渲染尺度的合同，不据此断言provider替换能修复。
- 任务15（必要性能）：600.61秒正常结束，峰值12.248GiB且内存门通过；无超门或无法结束证据，本轮不启动。

## 6. 范围化结论与交付

- 当前等级：未达D1（几何门拒绝，无预览）；未达D2–D5。scale=0.25不能授予D2/D3；无用户D4确认，也不授予D5。
- 输入/参考保持：见analysis checks；45张均解码并登记15机位，输入SHA/参考/二进制检查通过；prepared_station_input不代表最终成片像素。无最终合成，未凭空排除源图。
- 本卡仅覆盖本次全量清单与冻结二进制；不挪用10张子集通过结论，也不外推scale=1、其他线程数或并发场景。
- 复看：诊断结果未提交用户画质确认，队列维持0/3；预览不是已验收成片。
- L0结论：接受本轮摸底诊断与上方区域归组；修复按既有批次/依赖排队，保持全部拒绝及观测缺口，不授予原生或用户画质验收。
- 下一步：任务16进入站点求解组，保留本次失败；已有9张scale=1通过仅保护该子集，不能覆盖这45张全量。
