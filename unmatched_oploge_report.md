# 无法整改的 OP_LOGE 报告

## 统计信息
- 扫描总数: 181 (所有18个算子的OP_LOGE调用)
- 已整改: 59 (已替换为 OP_LOGE_FOR_* 系列宏)
- 无法整改: 122
- 推荐新增错误码: 12

## 无法整改列表

| # | 算子 | 文件路径 | 行号 | 原错误消息 | 分类 | 推荐新增错误码 |
|---|------|----------|------|------------|------|----------------|
| 1 | TruncatedNormal | random/truncated_normal_v2/op_host/arch35/truncated_normal_v2_tiling_arch35.cpp | 79 | totalCoreNum is less than or equal to 0. please check. | Platform Config | EZ0040_PLATFORM_CONFIG_INVALID |
| 2 | TruncatedNormal | random/truncated_normal_v2/op_graph/fusion_pass/truncated_normal_fusion_pass.cpp | 90 | Get platformInfo failed. | Fusion Pass Error | EZ0101_FUSION_PLATFORM_INFO_FAILED |
| 3 | TruncatedNormal | random/truncated_normal_v2/op_graph/fusion_pass/truncated_normal_fusion_pass.cpp | 101 | Failed to GetCaptured tensor. | Fusion Pass Error | EZ0102_FUSION_GET_CAPTURED_TENSOR_FAILED |
| 4 | TruncatedNormal | random/truncated_normal_v2/op_graph/fusion_pass/truncated_normal_fusion_pass.cpp | 135 | Failed to GetCaptured tensor in Replacement. | Fusion Pass Error | EZ0102_FUSION_GET_CAPTURED_TENSOR_FAILED |
| 5 | StatelessRandperm | random/stateless_randperm/op_host/arch35/stateless_randperm_tiling_arch35.cpp | 65 | UB size is invalid. | Platform Config | EZ0040_PLATFORM_CONFIG_INVALID |
| 6 | StatelessRandperm | random/stateless_randperm/op_host/arch35/stateless_randperm_tiling_arch35.cpp | 68 | UB size %ld bytes must be greater than simt dcache size %ld bytes, please check. | Platform Config | EZ0040_PLATFORM_CONFIG_INVALID |
| 7 | StatelessRandperm | random/stateless_randperm/op_host/arch35/stateless_randperm_tiling_arch35.cpp | 134 | max value of [attr]dtype not found in. | Internal Consistency | EZ0041_MAP_LOOKUP_FAILED |
| 8 | StatelessRandperm | random/stateless_randperm/op_host/arch35/stateless_randperm_tiling_arch35.cpp | 253 | GetAttrs failed! | Tiling Internal Failure | EZ0035_TILING_GET_ATTRS_FAILED |
| 9 | StatelessRandperm | random/stateless_randperm/op_host/arch35/stateless_randperm_tiling_arch35.cpp | 254 | GetInputN failed! | Tiling Internal Failure | EZ0036_TILING_GET_INPUT_FAILED |
| 10 | StatelessRandperm | random/stateless_randperm/op_host/arch35/stateless_randperm_tiling_arch35.cpp | 255 | GetInputSeed failed! | Tiling Internal Failure | EZ0036_TILING_GET_INPUT_FAILED |
| 11 | StatelessRandperm | random/stateless_randperm/op_host/arch35/stateless_randperm_tiling_arch35.cpp | 257 | GetInputOffset failed! | Tiling Internal Failure | EZ0036_TILING_GET_INPUT_FAILED |
| 12 | StatelessRandperm | random/stateless_randperm/op_host/arch35/stateless_randperm_tiling_arch35.cpp | 258 | GetOutputY failed! | Tiling Internal Failure | EZ0037_TILING_GET_OUTPUT_FAILED |
| 13 | StatelessRandperm | random/stateless_randperm/op_host/arch35/stateless_randperm_tiling_arch35.cpp | 324 | tilingBuilder failed. | Tiling Internal Failure | EZ0038_TILING_BUILDER_FAILED |
| 14 | StatelessRandperm | random/stateless_randperm/op_host/arch35/stateless_randperm_tiling_arch35.cpp | 326 | SortTilingSimt failed. | Tiling Internal Failure | EZ0038_TILING_BUILDER_FAILED |
| 15 | StatelessRandperm | random/stateless_randperm/op_host/arch35/stateless_randperm_tiling_arch35.cpp | 329 | sortTilingData is nullptr. | Internal Nullptr | EZ0042_TILING_DATA_NULLPTR |
| 16 | StatelessRandperm | random/stateless_randperm/op_host/arch35/stateless_randperm_tiling_arch35.cpp | 333 | userWorkSpaceSize is nullptr. | Internal Nullptr | EZ0042_TILING_DATA_NULLPTR |
| 17 | StatelessRandperm | random/stateless_randperm/op_host/arch35/stateless_randperm_tiling_arch35.cpp | 409 | SortTilingBridge failed. | Tiling Internal Failure | EZ0038_TILING_BUILDER_FAILED |
| 18 | StatelessRandperm | random/stateless_randperm/op_host/arch35/stateless_randperm_tiling_arch35.cpp | 516 | Tiling context is nullptr! | Internal Nullptr | EZ0042_TILING_DATA_NULLPTR |
| 19 | StatelessRandperm | random/stateless_randperm/op_host/arch35/stateless_randperm_tiling_arch35.cpp | 530 | Tiling context is nullptr! | Internal Nullptr | EZ0042_TILING_DATA_NULLPTR |
| 20 | StatelessRandperm | random/stateless_randperm/op_host/arch35/stateless_randperm_tiling_for_sort.cpp | 105 | allUb must greater than simtUb, but is %lu | Platform Config | EZ0040_PLATFORM_CONFIG_INVALID |
| 21 | StatelessRandperm | random/stateless_randperm/op_host/arch35/stateless_randperm_tiling_for_sort.cpp | 148 | CheckInputAndOutput failed | Tiling Internal Failure | EZ0039_TILING_CHECK_PARAMS_FAILED |
| 22 | StatelessRandperm | random/stateless_randperm/op_host/arch35/stateless_randperm_tiling_for_sort.cpp | 545 | alignNum is 0 | Tiling Internal Failure | EZ0039_TILING_CHECK_PARAMS_FAILED |
| 23 | StatelessRandperm | random/stateless_randperm/op_host/arch35/stateless_randperm_tiling_for_sort.cpp | 589 | get tilingdata ptr failed | Internal Nullptr | EZ0042_TILING_DATA_NULLPTR |
| 24 | StatelessRandperm | random/stateless_randperm/op_host/arch35/stateless_randperm_tiling_for_sort.cpp | 593 | memset tilingdata failed | Tiling Internal Failure | EZ0039_TILING_CHECK_PARAMS_FAILED |
| 25 | StatelessRandperm | random/stateless_randperm/op_host/arch35/stateless_randperm_tiling_for_sort.cpp | 597 | check params failed | Tiling Internal Failure | EZ0039_TILING_CHECK_PARAMS_FAILED |
| 26 | DropOutV3 | random/drop_out_v3/op_host/arch35/drop_out_v3_tiling_arch35.cpp | 60 | get seed value failed | Tiling Internal Failure | EZ0036_TILING_GET_INPUT_FAILED |
| 27 | DropOutV3 | random/drop_out_v3/op_host/arch35/drop_out_v3_tiling_arch35.cpp | 65 | get offset value failed | Tiling Internal Failure | EZ0036_TILING_GET_INPUT_FAILED |
| 28 | DropOutV3 | random/drop_out_v3/op_host/arch35/drop_out_v3_tiling_arch35.cpp | 88 | get const shape of prob failed | Tiling Internal Failure | EZ0036_TILING_GET_INPUT_FAILED |
| 29 | DropOutV3 | random/drop_out_v3/op_api/aclnn_dropout_v3.cpp | 64 | currently, the input of noise_shape must be nullptr, please check. | op_api ACLNN Error | - |
| 30 | DropOutV3 | random/drop_out_v3/op_api/aclnn_dropout_v3.cpp | 104 | The value of p has to be between 0 and 1, but current is %f. | op_api ACLNN Error | - |
| 31 | DropOutV3 | random/drop_out_v3/op_api/aclnn_dropout_v3.cpp | 119 | Size of maskOut has to be %ld, but current is %ld. | op_api ACLNN Error | - |
| 32 | DropOutV3 | random/drop_out_v3/op_api/aclnn_dropout_v3.cpp | 151 | The length of tuple returned by DropoutV3 is not 2. | op_api ACLNN Error | - |
| 33 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_split_fusion_pass.cpp | 63 | Get platform_info failed. | Fusion Pass Error | EZ0101_FUSION_PLATFORM_INFO_FAILED |
| 34 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_split_fusion_pass.cpp | 79 | x dtype only support float/float16/bf16, actual: %d | Fusion Pass Error | EZ0103_FUSION_DTYPE_CHECK_FAILED |
| 35 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_split_fusion_pass.cpp | 87 | p dtype only support float/float16/bf16, actual: %d | Fusion Pass Error | EZ0103_FUSION_DTYPE_CHECK_FAILED |
| 36 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_split_fusion_pass.cpp | 95 | seed dtype only support int32/int64, actual: %d | Fusion Pass Error | EZ0103_FUSION_DTYPE_CHECK_FAILED |
| 37 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_split_fusion_pass.cpp | 103 | y dtype only support float/float16/bf16, actual: %d | Fusion Pass Error | EZ0103_FUSION_DTYPE_CHECK_FAILED |
| 38 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_split_fusion_pass.cpp | 108 | x dtype should same with y dtype, x: %d, y: %d | Fusion Pass Error | EZ0103_FUSION_DTYPE_CHECK_FAILED |
| 39 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_split_fusion_pass.cpp | 128 | DropOutV3 input size != 5, actual: %zu | Fusion Pass Error | EZ0104_FUSION_INPUT_SIZE_FAILED |
| 40 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_split_fusion_pass.cpp | 140 | x dim should be [0~8], actual: %d | Fusion Pass Error | EZ0105_FUSION_SHAPE_CHECK_FAILED |
| 41 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_split_fusion_pass.cpp | 256 | AddInput failed for idx %zu | Fusion Pass Error | EZ0106_FUSION_ADD_INPUT_OUTPUT_FAILED |
| 42 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_split_fusion_pass.cpp | 263 | AddOutput failed for output 0 | Fusion Pass Error | EZ0106_FUSION_ADD_INPUT_OUTPUT_FAILED |
| 43 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_split_fusion_pass.cpp | 270 | AddOutput failed for output 1 | Fusion Pass Error | EZ0106_FUSION_ADD_INPUT_OUTPUT_FAILED |
| 44 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_split_fusion_pass.cpp | 300 | CreateReplacement failed for node %s | Fusion Pass Error | EZ0107_FUSION_CREATE_REPLACEMENT_FAILED |
| 45 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_split_fusion_pass.cpp | 309 | ConstructBoundary failed for node %s | Fusion Pass Error | EZ0108_FUSION_CONSTRUCT_BOUNDARY_FAILED |
| 46 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_split_fusion_pass.cpp | 318 | SubgraphRewriter::Replace failed for node %s, status=%d | Fusion Pass Error | EZ0109_FUSION_REPLACE_FAILED |
| 47 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_fusion_pass.cpp | 134 | GetCapturedTensor failed for index %ld | Fusion Pass Error | EZ0102_FUSION_GET_CAPTURED_TENSOR_FAILED |
| 48 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_fusion_pass.cpp | 216 | Get platform_info failed. | Fusion Pass Error | EZ0101_FUSION_PLATFORM_INFO_FAILED |
| 49 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_fusion_pass.cpp | 236 | Expected StatelessDropOutGenMask, got %s | Fusion Pass Error | EZ0103_FUSION_DTYPE_CHECK_FAILED |
| 50 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_fusion_pass.cpp | 241 | GenMask input size != 5 | Fusion Pass Error | EZ0104_FUSION_INPUT_SIZE_FAILED |
| 51 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_fusion_pass.cpp | 248 | GenMask prob dtype not supported | Fusion Pass Error | EZ0103_FUSION_DTYPE_CHECK_FAILED |
| 52 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_fusion_pass.cpp | 255 | GenMask seed dtype not supported | Fusion Pass Error | EZ0103_FUSION_DTYPE_CHECK_FAILED |
| 53 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_fusion_pass.cpp | 262 | GenMask offset dtype != DT_INT64 | Fusion Pass Error | EZ0103_FUSION_DTYPE_CHECK_FAILED |
| 54 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_fusion_pass.cpp | 269 | GenMask output dtype != DT_UINT8 | Fusion Pass Error | EZ0103_FUSION_DTYPE_CHECK_FAILED |
| 55 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_fusion_pass.cpp | 286 | Expected DropOutDoMask, got %s | Fusion Pass Error | EZ0103_FUSION_DTYPE_CHECK_FAILED |
| 56 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_fusion_pass.cpp | 291 | DoMask input size != 3 | Fusion Pass Error | EZ0104_FUSION_INPUT_SIZE_FAILED |
| 57 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_fusion_pass.cpp | 298 | DoMask x dtype not supported | Fusion Pass Error | EZ0103_FUSION_DTYPE_CHECK_FAILED |
| 58 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_fusion_pass.cpp | 305 | DoMask y dtype not supported | Fusion Pass Error | EZ0103_FUSION_DTYPE_CHECK_FAILED |
| 59 | DropOutV3 | random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_fusion_pass.cpp | 329 | Infershape failed. | Fusion Pass Error | EZ0110_FUSION_INFERSHAPE_FAILED |
| 60 | StatelessRandomChoiceWithMask | random/stateless_random_choice_with_mask/op_host/arch35/stateless_random_choice_with_mask_simt_tiling.cpp | 34 | compile info is null | Tiling Internal Failure | EZ0042_TILING_DATA_NULLPTR |
| 61 | StatelessRandomChoiceWithMask | random/stateless_random_choice_with_mask/op_host/arch35/stateless_random_choice_with_mask_simt_tiling.cpp | 48 | coreNum and ubSize should not be samller than 0, but got coreNum [%ld] and ubSize [%ld] | Platform Config | EZ0040_PLATFORM_CONFIG_INVALID |
| 62 | StatelessBernoulli | random/stateless_bernoulli/op_host/arch35/stateless_bernoulli_tiling_arch35.cpp | 84 | totalCoreNum is less than or equal to 0. please check. | Platform Config | EZ0040_PLATFORM_CONFIG_INVALID |
| 63 | StatelessBernoulli | random/stateless_bernoulli/op_api/stateless_bernoulli.cpp | 85 | StatelessBernoulli inputShape is null. | ACLNN_ERR_INNER_NULLPTR Error | - |
| 64 | StatelessBernoulli | random/stateless_bernoulli/op_graph/fusion_pass/stateless_bernoulli_fusion_pass.cpp | 101 | Get platformInfo failed. | Fusion Pass Error | EZ0101_FUSION_PLATFORM_INFO_FAILED |
| 65 | StatelessBernoulli | random/stateless_bernoulli/op_graph/fusion_pass/stateless_bernoulli_fusion_pass.cpp | 110 | Failed to GetCaptured tensor | Fusion Pass Error | EZ0102_FUSION_GET_CAPTURED_TENSOR_FAILED |
| 66 | StatelessBernoulli | random/stateless_bernoulli/op_graph/fusion_pass/stateless_bernoulli_fusion_pass.cpp | 140 | Failed to GetCaptured tensor in Replacement | Fusion Pass Error | EZ0102_FUSION_GET_CAPTURED_TENSOR_FAILED |
| 67 | StatelessBernoulli | random/stateless_bernoulli/op_graph/fusion_pass/stateless_bernoulli_fusion_pass.cpp | 183 | Infershape failed. | Fusion Pass Error | EZ0110_FUSION_INFERSHAPE_FAILED |
| 68 | DropOutDoMask | random/drop_out_do_mask/op_host/arch35/drop_out_do_mask_tiling_arch35.cpp | 107 | input shape check failed. | Tiling Internal Failure | EZ0039_TILING_CHECK_PARAMS_FAILED |
| 69 | DropOutDoMask | random/drop_out_do_mask/op_host/arch35/drop_out_do_mask_tiling_arch35.cpp | 115 | get dataType size fail. | Tiling Internal Failure | EZ0039_TILING_CHECK_PARAMS_FAILED |
| 70 | DropOutDoMask | random/drop_out_do_mask/op_host/arch35/drop_out_do_mask_tiling.cpp | 46 | Failed to get core num. | Platform Config | EZ0040_PLATFORM_CONFIG_INVALID |
| 71 | DropOutDoMask | random/drop_out_do_mask/op_host/arch35/drop_out_do_mask_tiling.cpp | 52 | Failed to get ub size. | Platform Config | EZ0040_PLATFORM_CONFIG_INVALID |
| 72 | DropOutDoMask | random/drop_out_do_mask/op_api/dropout_do_mask.cpp | 62 | DropoutDoMaskAiCore ADD_TO_LAUNCHER_LIST_AICORE failed. | ACLNN_ERR_INNER_NULLPTR Error | - |
| 73 | DropOutDoMask | random/drop_out_do_mask/op_api/dropout_do_mask.cpp | 76 | DropoutDoMaskAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed. | ACLNN_ERR_INNER_NULLPTR Error | - |
| 74 | DropOutDoMask | random/drop_out_do_mask/op_api/aclnn_dropout_backward.cpp | 97 | The value of scale is error, p = (scale == 0.0) ? 1 : (1 - 1 / scale) has to be between 0 and 1, but got scale %f. | op_api ACLNN Error | - |
| 75 | DropOutDoMask | random/drop_out_do_mask/op_api/aclnn_dropout_backward.cpp | 116 | Size of maskOut has to be %ld, but current is %ld. | op_api ACLNN Error | - |
| 76 | DropOutDoMask | random/drop_out_do_mask/op_api/aclnn_dropout_do_mask.cpp | 95 | The value of prob has to be between 0 and 1, but current is %f. | op_api ACLNN Error | - |
| 77 | DropOutDoMask | random/drop_out_do_mask/op_api/aclnn_dropout_do_mask.cpp | 110 | Size of mask has to be %ld, but current is %ld. | op_api ACLNN Error | - |
| 78 | TensorEqual | math/tensor_equal/op_host/arch35/tensor_equal_tiling_arch35.cpp | 51 | ub size is invalid. | Platform Config | EZ0040_PLATFORM_CONFIG_INVALID |
| 79 | TensorEqual | math/tensor_equal/op_host/arch35/tensor_equal_tiling_arch35.cpp | 201 | SetBlockDim failed. | Tiling Internal Failure | EZ0039_TILING_CHECK_PARAMS_FAILED |
| 80 | TensorEqual | math/tensor_equal/op_host/arch35/tensor_equal_tiling_arch35.cpp | 245 | The context is nullptr. | Internal Nullptr | EZ0042_TILING_DATA_NULLPTR |
| 81 | TensorEqual | math/tensor_equal/op_host/arch35/tensor_equal_tiling_arch35.cpp | 260 | The context is nullptr. | Internal Nullptr | EZ0042_TILING_DATA_NULLPTR |
| 82 | TensorEqual | math/tensor_equal/op_host/arch35/tensor_equal_tiling_arch35.cpp | 276 | Failed to get ub size | Platform Config | EZ0040_PLATFORM_CONFIG_INVALID |
| 83 | TensorEqual | math/tensor_equal/op_api/aclnn_equal.cpp | 85 | Self dtype %s and other dtype %s can not promote dtype. | op_api ACLNN Error | - |
| 84 | TensorEqual | math/tensor_equal/op_api/aclnn_equal.cpp | 92 | Self dtype %s and other dtype %s get promoteType dtype %s should be in dtype support list [%s]. | op_api ACLNN Error | - |
| 85 | RandomStandardNormal | random/random_standard_normal_v2/op_host/arch35/random_standard_normal_v2_tiling_arch35.cpp | 83 | RandomStandardNormalV2 GetHardwareInfo Failed, vectorCoreNum:%ld, ubSize:%ld. | Platform Config | EZ0040_PLATFORM_CONFIG_INVALID |
| 86 | RandomStandardNormal | random/random_standard_normal_v2/op_graph/fusion_pass/random_standard_normal_fusion_pass.cpp | 101 | Get platform info failed. | Fusion Pass Error | EZ0101_FUSION_PLATFORM_INFO_FAILED |
| 87 | RandomStandardNormal | random/random_standard_normal_v2/op_graph/fusion_pass/random_standard_normal_fusion_pass.cpp | 113 | Failed to get captured tensor. | Fusion Pass Error | EZ0102_FUSION_GET_CAPTURED_TENSOR_FAILED |
| 88 | RandomStandardNormal | random/random_standard_normal_v2/op_graph/fusion_pass/random_standard_normal_fusion_pass.cpp | 141 | SubgraphInputs is empty, cannot get input info. | Fusion Pass Error | EZ0102_FUSION_GET_CAPTURED_TENSOR_FAILED |
| 89 | RandomStandardNormal | random/random_standard_normal_v2/op_graph/fusion_pass/random_standard_normal_fusion_pass.cpp | 152 | Failed to GetCaptured tensor in Replacement. | Fusion Pass Error | EZ0102_FUSION_GET_CAPTURED_TENSOR_FAILED |
| 90 | RandomStandardNormal | random/random_standard_normal_v2/op_graph/fusion_pass/random_standard_normal_fusion_pass.cpp | 174 | Failed to get node name. | Fusion Pass Error | EZ0102_FUSION_GET_CAPTURED_TENSOR_FAILED |
| 91 | StatelessDropOutGenMask | random/stateless_drop_out_gen_mask/op_host/arch35/stateless_drop_out_gen_mask_tiling_arch35.cpp | 67 | GetOutputSize failed | Tiling Internal Failure | EZ0037_TILING_GET_OUTPUT_FAILED |
| 92 | StatelessRandomUniformV2 | random/stateless_random_uniform_v2/op_host/arch35/stateless_random_uniform_v2_tiling_arch35.cpp | 27 | compile info is null | Tiling Internal Failure | EZ0042_TILING_DATA_NULLPTR |
| 93 | StatelessRandomUniformV2 | random/stateless_random_uniform_v2/op_host/arch35/stateless_random_uniform_v2_tiling_arch35.cpp | 29 | StatelessRandomUniformV2Tiling fail to get coreNum. | Platform Config | EZ0040_PLATFORM_CONFIG_INVALID |
| 94 | StatelessRandomUniformV2 | random/stateless_random_uniform_v2/op_host/arch35/stateless_random_uniform_v2_tiling_arch35.cpp | 32 | ub size less than 0 Size. please check | Platform Config | EZ0040_PLATFORM_CONFIG_INVALID |
| 95 | StatelessRandomNormalV2 | random/stateless_random_normal_v2/op_host/arch35/stateless_random_normal_v2_tiling_arch35.cpp | 39 | compile info is null | Tiling Internal Failure | EZ0042_TILING_DATA_NULLPTR |
| 96 | StatelessRandomNormalV2 | random/stateless_random_normal_v2/op_host/arch35/stateless_random_normal_v2_tiling_arch35.cpp | 46 | StatelessRandomNormalV2Tiling fail to get coreNum. | Platform Config | EZ0040_PLATFORM_CONFIG_INVALID |
| 97 | StatelessRandomNormalV2 | random/stateless_random_normal_v2/op_host/arch35/stateless_random_normal_v2_tiling_arch35.cpp | 53 | ub size less than REGBASE_CCEC_CACHE_SIZE Size. please check | Platform Config | EZ0040_PLATFORM_CONFIG_INVALID |
| 98 | StatelessRandomNormalV2 | random/stateless_random_normal_v2/op_api/aclnn_normal_out.cpp | 61 | Shape of out should be %s, but current is %s. | op_api ACLNN Error | - |
| 99 | StatelessRandomNormalV2 | random/stateless_random_normal_v2/op_api/aclnn_normal_out.cpp | 88 | The std can not be an empty tensor. | op_api ACLNN Error | - |
| 100 | StatelessRandomNormalV2 | random/stateless_random_normal_v2/op_api/aclnn_normal_out.cpp | 93 | Shape of std should be match with out. | op_api ACLNN Error | - |
| 101 | StatelessRandomNormalV2 | random/stateless_random_normal_v2/op_api/aclnn_normal_out.cpp | 130 | Shape of out should be %s, but current is %s. | op_api ACLNN Error | - |
| 102 | StatelessRandomNormalV2 | random/stateless_random_normal_v2/op_api/aclnn_normal_out.cpp | 150 | Mean dtype [%s] and std dtype [%s] can not promote dtype. | op_api ACLNN Error | - |
| 103 | DSARandomUniform | random/dsa_random_uniform/op_host/op_api/aclnn_random.cpp | 209 | from value %ld is out of valid range [%ld, %ld] for dtype %d | op_api ACLNN Error | - |
| 104 | DSARandomUniform | random/dsa_random_uniform/op_host/op_api/aclnn_random.cpp | 215 | to value %ld is out of valid range (%ld, %lu] for dtype %d | op_api ACLNN Error | - |
| 105 | DSARandomUniform | random/dsa_random_uniform/op_host/op_api/aclnn_random.cpp | 221 | from %ld must be less than to %ld. | op_api ACLNN Error | - |
| 106 | DSARandomUniform | random/dsa_random_uniform/op_host/op_api/aclnn_random.cpp | 323 | Random seedTensor dtype must be DT_INT64, got %s. | op_api ACLNN Error | - |
| 107 | DSARandomUniform | random/dsa_random_uniform/op_host/op_api/aclnn_random.cpp | 328 | Random offsetTensor dtype must be DT_INT64, got %s. | op_api ACLNN Error | - |
| 108 | DSARandomUniform | random/dsa_random_uniform/op_host/op_api/aclnn_random.cpp | 569 | Random seedTensor dtype must be DT_INT64, got %s. | op_api ACLNN Error | - |
| 109 | DSARandomUniform | random/dsa_random_uniform/op_host/op_api/aclnn_random.cpp | 574 | Random offsetTensor dtype must be DT_INT64, got %s. | op_api ACLNN Error | - |
| 110 | DSARandomUniform | random/dsa_random_uniform/op_host/op_api/aclnn_uniform.cpp | 68 | Input dtype of aclnnInplaceUniform is not support bfloat16 in current socversion. | op_api ACLNN Error | - |
| 111 | DSARandomUniform | random/dsa_random_uniform/op_host/op_api/aclnn_uniform.cpp | 134 | from value %lf is out of valid range [%lf, %lf] for dtype %d | op_api ACLNN Error | - |
| 112 | DSARandomUniform | random/dsa_random_uniform/op_host/op_api/aclnn_uniform.cpp | 140 | to value %lf is out of valid range [%lf, %lf] for dtype %d | op_api ACLNN Error | - |
| 113 | DSARandomUniform | random/dsa_random_uniform/op_host/op_api/aclnn_uniform.cpp | 146 | from %lf must be less than or equal to %lf. | op_api ACLNN Error | - |
| 114 | DSARandomUniform | random/dsa_random_uniform/op_host/op_api/aclnn_uniform.cpp | 151 | to-from exceeds the limit of dtype %d, to=%lf, from=%lf. | op_api ACLNN Error | - |
| 115 | DSARandomUniform | random/dsa_random_uniform/op_host/op_api/aclnn_uniform.cpp | 180 | invalid dtype, must be bfloat16 or float16. | op_api ACLNN Error | - |
| 116 | DSARandomUniform | random/dsa_random_uniform/op_host/op_api/aclnn_multinomial.cpp | 117 | Dim of self only can be 1 or 2. | op_api ACLNN Error | - |
| 117 | DSARandomUniform | random/dsa_random_uniform/op_host/op_api/aclnn_multinomial.cpp | 121 | Dim of self should be equal to dim of out. | op_api ACLNN Error | - |
| 118 | DSARandomUniform | random/dsa_random_uniform/op_host/op_api/aclnn_multinomial.cpp | 127 | excepted the size of out at last dim must be equal with numsamples %ld, but got %ld. | op_api ACLNN Error | - |
| 119 | DSARandomUniform | random/dsa_random_uniform/op_host/op_api/aclnn_multinomial.cpp | 131 | excepted the size of out at first dim %ld must be equal with the size of self at first dim %ld when dimNum > 1 | op_api ACLNN Error | - |
| 120 | DSARandomUniform | random/dsa_random_uniform/op_host/op_api/aclnn_multinomial.cpp | 140 | Numsamples must > 0. | op_api ACLNN Error | - |
| 121 | DSARandomUniform | random/dsa_random_uniform/op_host/op_api/aclnn_multinomial.cpp | 146 | numsamples must <= shape.GetDim(dimNum - 1) without replacement | op_api ACLNN Error | - |
| 122 | DSARandomUniform | random/dsa_random_uniform/op_host/op_api/aclnn_multinomial.cpp | 150 | number of categories cannot exceed 2^24 | op_api ACLNN Error | - |
| 123 | SimThreadExponential | random/sim_thread_exponential/op_host/arch32/sim_thread_exponential_tiling_arch32.cpp | 140 | ThreadPerProcessor %d must be greater than 0. | Platform Config | EZ0040_PLATFORM_CONFIG_INVALID |
| 124 | SimThreadExponential | random/sim_thread_exponential/op_host/arch32/sim_thread_exponential_tiling_arch32.cpp | 144 | StreamProcessorCount %d must be greater than 0. | Platform Config | EZ0040_PLATFORM_CONFIG_INVALID |
| 125 | SimThreadExponential | random/sim_thread_exponential/op_host/arch32/sim_thread_exponential_tiling_arch32.cpp | 148 | Start %f must be less than or equal to end %f. | Tiling Internal Failure | EZ0039_TILING_CHECK_PARAMS_FAILED |
| 126 | SimThreadExponential | random/sim_thread_exponential/op_host/arch32/sim_thread_exponential_tiling_arch32.cpp | 215 | batchNumPerCore %u must be greater than 0. | Tiling Internal Failure | EZ0039_TILING_CHECK_PARAMS_FAILED |
| 127 | SimThreadExponential | random/sim_thread_exponential/op_host/arch32/sim_thread_exponential_tiling_arch32.cpp | 267 | SetAttrParams failed. | Tiling Internal Failure | EZ0035_TILING_GET_ATTRS_FAILED |
| 128 | SimThreadExponential | random/sim_thread_exponential/op_host/arch32/sim_thread_exponential_tiling_arch32.cpp | 269 | GetInputTensorInfo failed. | Tiling Internal Failure | EZ0036_TILING_GET_INPUT_FAILED |
| 129 | SimThreadExponential | random/sim_thread_exponential/op_host/arch32/sim_thread_exponential_tiling_arch32.cpp | 272 | GetPlatformInfo failed. | Tiling Internal Failure | EZ0040_PLATFORM_CONFIG_INVALID |
| 130 | SimThreadExponential | random/sim_thread_exponential/op_host/arch32/sim_thread_exponential_tiling_arch32.cpp | 274 | Tiling4Block failed. | Tiling Internal Failure | EZ0038_TILING_BUILDER_FAILED |

> **注**: DropOutDoMaskV3 (2条OP_LOGE_FOR_*)、DropOutDoMaskV3d (0条)、StatelessRandom (4条OP_LOGE_FOR_*)、StatelessUniform (0条)、StatelessNormal (2条OP_LOGE_FOR_*) 已全部整改完毕，无无法整改项。

## 分类统计

| 分类 | 数量 | 说明 |
|------|------|------|
| Fusion Pass Error | 39 | fusion_pass基础设施错误（GetCapturedTensor/platform_info/Infershape/AddInput/AddOutput/CreateReplacement/ConstructBoundary/Replace/dtype检查/input size/节点类型等），使用OP_LOGE(kPassName.c_str(), ...)格式 |
| Tiling Internal Failure | 28 | Tiling内部流程错误（GetAttrs/GetInput*/GetOutput*/tilingBuilder/SortTiling*/check params/memset/SetBlockDim/GetPlatformInfo/Tiling4Block等），使用OP_LOGE(opName_, ...)或OP_LOGE(context->GetNodeName(), ...)格式 |
| Platform/Compile Config | 17 | 平台编译配置无效（coreNum<=0/ubSize<=0/ubSize<=SIMT_DCACHE_SIZE/ubSize<=REGBASE_CCEC_CACHE_SIZE/ubSize invalid/dtype size fail等） |
| op_api ACLNN Error | 38 | op_api层使用ACLNN_ERR_PARAM_INVALID/ACLNN_ERR_INNER_NULLPTR的OP_LOGE，采用opdev/op_log.h框架，与EZ宏体系不兼容 |
| ACLNN_ERR_INNER_NULLPTR Error | 3 | op_api层内部空指针/launcher失败（inputShape null/ADD_TO_LAUNCHER_LIST failed） |
| Internal Nullptr | 8 | Tiling层内部空指针（context nullptr/compileInfo null/sortTilingData nullptr/userWorkSpaceSize nullptr/tilingdata ptr failed） |
| Internal Consistency | 1 | maxValueMap查找失败 |

## 无法整改原因说明

### 1. Fusion Pass Error (39条)
- **原因**: fusion_pass代码使用`OP_LOGE(kPassName.c_str(), "...")`格式，属于图融合基础设施层。EZ0008-EZ0034系列宏设计针对通用算子参数验证，不适用于融合pass内部流程错误。融合pass是GE图优化阶段的内部组件，其错误不影响算子参数合法性，而是影响图改写流程。
- **整改建议**: 需新增EZ0101-EZ0110系列融合pass专用错误码宏。

### 2. Tiling Internal Failure (28条)
- **原因**: 这些OP_LOGE用于tiling内部流程（GetAttrs/GetInput*/GetOutput*/tilingBuilder等），属于Tiling阶段的内部计算流程错误，不属于参数验证范畴。例如"GetAttrs failed"是tiling内部获取属性失败，而非用户输入参数不合法。
- **整改建议**: 需新增EZ0035-EZ0039系列Tiling内部流程专用错误码宏。

### 3. Platform/Compile Config (17条)
- **原因**: 这些OP_LOGE检查的是从编译信息/平台信息中获取的硬件配置值（coreNum、ubSize等）是否合法。这些值来源于编译阶段的平台信息，而非用户输入参数，EZ宏中的"INVALID"系列针对用户输入参数验证，不适用于平台配置内部错误。
- **整改建议**: 需新增EZ0040平台配置专用错误码宏。

### 4. op_api ACLNN Error (38条)
- **原因**: 这些OP_LOGE使用`opdev/op_log.h`框架的`ACLNN_ERR_PARAM_INVALID`/`ACLNN_ERR_INNER_NULLPTR`作为第一个参数。op_api层的OP_LOGE采用的是完全不同的日志框架（opdev/op_log.h vs opbase/log.h），EZ宏定义在opbase/log.h中，两者不兼容。op_api层错误码体系使用aclnnStatus（161xxx/361xxx/561xxx系列），与EZ0008-EZ0034的宏体系架构不同。
- **整改建议**: 无需整改，op_api层保持现有ACLNN_ERR_PARAM_INVALID/ACLNN_ERR_INNER_NULLPTR格式即可。

### 5. ACLNN_ERR_INNER_NULLPTR Error (3条)
- **原因**: 同op_api ACLNN Error，使用opdev/op_log.h框架。
- **整改建议**: 无需整改。

### 6. Internal Nullptr (8条)
- **原因**: 这些OP_LOGE检查的是tiling阶段内部对象的空指针（context、compileInfo、sortTilingData、userWorkSpaceSize等），属于内部流程保护性检查，不是用户输入参数验证。EZ0009_NULLPTR宏针对的是算子输入输出tensor空指针验证。
- **整改建议**: 需新增EZ0042_TILING_DATA_NULLPTR宏。

### 7. Internal Consistency (1条)
- **原因**: maxValueMap查找失败属于内部映射表查找逻辑错误，不属于参数验证范畴。
- **整改建议**: 需新增EZ0041_MAP_LOOKUP_FAILED宏。

## 推荐新增错误码汇总

| 推荐错误码 | 场景 | ErrMessage 建议 | 出现次数 |
|------------|------|-----------------|----------|
| EZ0035_TILING_GET_ATTRS_FAILED | Tiling获取属性失败 | "GetAttrs failed in tiling phase." | 2 |
| EZ0036_TILING_GET_INPUT_FAILED | Tiling获取输入信息失败 | "GetInput* failed in tiling phase." | 4 |
| EZ0037_TILING_GET_OUTPUT_FAILED | Tiling获取输出信息失败 | "GetOutput* failed in tiling phase." | 2 |
| EZ0038_TILING_BUILDER_FAILED | Tiling构建器/桥接失败 | "Tiling builder or bridge failed." | 4 |
| EZ0039_TILING_CHECK_PARAMS_FAILED | Tiling内部参数检查失败 | "Tiling internal check params failed." | 8 |
| EZ0040_PLATFORM_CONFIG_INVALID | 平台/编译配置无效 | "Platform config value is invalid." | 17 |
| EZ0041_MAP_LOOKUP_FAILED | 内部映射表查找失败 | "Map lookup failed for key." | 1 |
| EZ0042_TILING_DATA_NULLPTR | Tiling数据空指针 | "Tiling internal data is nullptr." | 8 |
| EZ0101_FUSION_PLATFORM_INFO_FAILED | 融合pass获取平台信息失败 | "Fusion pass failed to get platform info." | 4 |
| EZ0102_FUSION_GET_CAPTURED_TENSOR_FAILED | 融合pass获取捕获tensor失败 | "Fusion pass failed to get captured tensor." | 8 |
| EZ0103_FUSION_DTYPE_CHECK_FAILED | 融合pass数据类型检查失败 | "Fusion pass dtype check failed." | 10 |
| EZ0104_FUSION_INPUT_SIZE_FAILED | 融合pass输入数量检查失败 | "Fusion pass input size check failed." | 2 |
| EZ0105_FUSION_SHAPE_CHECK_FAILED | 融合pass维度检查失败 | "Fusion pass shape dim check failed." | 1 |
| EZ0106_FUSION_ADD_INPUT_OUTPUT_FAILED | 融合pass添加输入输出失败 | "Fusion pass AddInput/AddOutput failed." | 3 |
| EZ0107_FUSION_CREATE_REPLACEMENT_FAILED | 融合pass创建替换图失败 | "Fusion pass CreateReplacement failed." | 1 |
| EZ0108_FUSION_CONSTRUCT_BOUNDARY_FAILED | 融合pass构建边界失败 | "Fusion pass ConstructBoundary failed." | 1 |
| EZ0109_FUSION_REPLACE_FAILED | 融合pass图改写失败 | "Fusion pass SubgraphRewriter::Replace failed." | 1 |
| EZ0110_FUSION_INFERSHAPE_FAILED | 融合pass形状推导失败 | "Fusion pass Infershape failed." | 2 |