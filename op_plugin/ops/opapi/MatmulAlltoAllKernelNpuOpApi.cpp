// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 *
 * Tensor -> const at::Tensor&
 * str -> c10::string_view
 * int -> int64_t
 * Tensor? -> const c10::optional<at::Tensor>
 * str? -> c10::optional<c10::string_view>
 * int? -> c10::optional<int64_t>
 *
 * Tensor -> std::vector<at::Tensor>
 *
 *
 * using tensor_list = std::tuple<at::Tensor, at::Tensor>;
 * (Tensor, Tensor) -> tensor_list
 *
 */

#include <set>
#include <op_plugin/OpApiInterface.h>
#include <torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h>
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpUtils.h"

static const size_t GROUP_DIM_1 = 1;
static const size_t GROUP_DIM_2 = 2;
static const uint64_t GROUP_MAX = 65535UL;
static const int DIM_TWO = 2;

namespace op_api
{
    using npu_preparation = at_npu::native::OpPreparation;

    // x valid dtype
    const std::set<int64_t> SUPPORT_X_DTYPE_LIST{
        static_cast<int64_t>(c10_npu::DType::FLOAT16),
        static_cast<int64_t>(c10_npu::DType::BFLOAT16),
        static_cast<int64_t>(c10_npu::DType::FLOAT8_E5M2),
        static_cast<int64_t>(c10_npu::DType::FLOAT8_E4M3FN)
    };
    // scales valid dtype
    const std::set<int64_t> SUPPORT_SCALES_DTYPE_LIST{
        static_cast<int64_t>(c10_npu::DType::FLOAT),
        static_cast<int64_t>(c10_npu::DType::FLOAT8_E8M0)
    };
    // output valid dtype
    const std::set<int64_t> SUPPORT_Y_DTYPE_LIST{
        static_cast<int64_t>(c10_npu::DType::FLOAT16),
        static_cast<int64_t>(c10_npu::DType::BFLOAT16),
        static_cast<int64_t>(c10_npu::DType::FLOAT)
    };

    bool is_valid_axes(const std::vector<int64_t>& a)
    {
        const std::vector<int64_t> expected = {-2, -1};
        return a == expected;
    }

    // int64_t check_and_get_groups(at::IntArrayRef group_size_list)
    // {
    //     int64_t groups = 0;
    //     if (group_size_list.empty()) {
    //         return groups;
    //     }
    //     size_t group_dim = group_size_list.size();
    //     TORCH_CHECK(group_dim == GROUP_DIM_1 || group_dim == GROUP_DIM_2, "group_sizes only support input with one or two elements, but got ",
    //                 group_dim, OPS_ERROR(ErrCode::PARAM));
    //     if (group_dim == GROUP_DIM_1) {
    //         int64_t group_bs = static_cast<int64_t>(group_size_list[0]);
    //         TORCH_CHECK((group_bs <= GROUP_MAX && group_m >= 0), "group param value must conform to range [0, 65535]", OPS_ERROR(ErrCode::VALUE));
    //         return groups
    //     }
    //     int64_t group_bs = static_cast<int64_t>(group_size_list[0]);
    //     int64_t group_h = static_cast<int64_t>(group_size_list[1]);
    //     bool invalid_group_param = ((group_bs <= GROUP_MAX && group_m >= 0)
    //                                 && (group_h <= GROUP_MAX && group_n >= 0));
    //     TORCH_CHECK(invalid_group_param, "group param value must conform to range [0, 65535]", OPS_ERROR(ErrCode::VALUE));
    //     return groups;
    // }

    at::Tensor npu_matmul_all_to_all(const at::Tensor &x1, const at::Tensor &x2, c10::string_view hcom, 
        const c10::optional<at::Tensor>& bias, const c10::optional<at::Tensor>& x1_scale, 
        const c10::optional<at::Tensor>& x2_scale, const c10::optional<at::Tensor>& common_scale, 
        const c10::optional<at::Tensor>& x1_offset, const c10::optional<at::Tensor>& x2_offset, 
        int64_t x1_quant_mode, int64_t x2_quant_mode, int64_t common_quant_mode, 
        int64_t group_sizes, c10::OptionalArrayRef all2all_axes,
        bool transpose_x1, bool transpose_x2, bool all2all_out_flag, 
        int64_t comm_quant_dtype, c10::optional<int64_t> x1_dtype, c10::optional<int64_t> x2_dtype, 
        c10::optional<int64_t> x1_scale_dtype, c10::optional<int64_t> x2_scale_dtype, 
        c10::optional<int64_t> output_scale_dtype, c10::optional<int64_t> comm_scale_dtype, c10::optional<int64_t> y_dtype
    )
    {
        // 校验x的shape, 2维(m, k) or (k, n)
        TORCH_CHECK(x1.dim() == DIM_TWO, "The x1 input of mm is required to be 2D, but the actual x1 input is ", x.dim(),
                    OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(x2.dim() == DIM_TWO, "The x2 input of mm is required to be 2D, but the actual x2 input is ", x.dim(),
                    OPS_ERROR(ErrCode::PARAM));

        // 校验x的dtype: float16/bfloat16/hifloat8/float8_e4m3fn/float8_e5m2
        if (x1_dtype.has_value())
        {
            TORCH_CHECK(SUPPORT_X_DTYPE_LIST.find(x1_dtype.value()) != SUPPORT_X_DTYPE_LIST.end(),
                        "The optional parameter x1_dtype only supports float16/bfloat16/hifloat8/float8_e4m3fn/float8_e5m2, but now is ",
                        op_plugin::utils::DTypeToString(x1_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
        }
        if (x2_dtype.has_value())
        {
            TORCH_CHECK(SUPPORT_X_DTYPE_LIST.find(x2_dtype.value()) != SUPPORT_X_DTYPE_LIST.end(),
                        "The optional parameter x2_dtype only supports float16/bfloat16/hifloat8/float8_e4m3fn/float8_e5m2, but now is ",
                        op_plugin::utils::DTypeToString(x2_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
        }
        // 校验scales的dtype: float/float8_e8m0
        if (x1_scale_dtype.has_value())
        {
            TORCH_CHECK(SUPPORT_SCALES_DTYPE_LIST.find(x1_scale_dtype.value()) != SUPPORT_SCALES_DTYPE_LIST.end(),
                        "The optional parameter x1_scale_dtype only supports float/float_e8m0, but now is ",
                        op_plugin::utils::DTypeToString(x1_scale_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
        }
        if (x2_scale_dtype.has_value())
        {
            TORCH_CHECK(SUPPORT_SCALES_DTYPE_LIST.find(x2_scale_dtype.value()) != SUPPORT_SCALES_DTYPE_LIST.end(),
                        "The optional parameter x2_scale_dtype only supports float/float_e8m0, but now is ",
                        op_plugin::utils::DTypeToString(x2_scale_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
        }

        // 校验all2all_axes，该值只能为空或者[-2,-1]
        std::vector<int64_t> all2all_axes_list = std::vector<int64_t>();
        if (all2all_axes.has_value())
        {
            all2all_axes_list = all2all_axes.value_or(at::IntArrayRef{});
            TORCH_CHECK(is_valid_axes(all2all_axes_list), "all2all_axes param value must be empty or [-2,-1]", OPS_ERROR(ErrCode::VALUE));
        }

        // 解析group_sizes
        // at::IntArrayRef group_size_list = {};
        // if (group_sizes.has_value()) {
        //     group_size_list = c10::asIntArrayRefUnchecked(group_sizes.value());
        // }
        // int64_t group_size = check_and_get_groups(group_size_list);
        // TORCH_CHECK(group_size != -1, "Invalid group_sizes.", OPS_ERROR(ErrCode::PARAM));

        // pta主要是为了推导output的shape和dtype，如果这里的output_dtype没有传入，则默认是bf16
        int64_t output_default_dtype = static_cast<int64_t>(at::ScalarType::BFloat16);
        if (y_dtype.has_value())
        {
            // 校验y的dtype: float16/bfloat16/float32
            TORCH_CHECK(SUPPORT_Y_DTYPE_LIST.find(y_dtype.value()) != SUPPORT_Y_DTYPE_LIST.end(),
                        "The optional parameter y_dtype only supports float16/bfloat16/float32, but now is ",
                        op_plugin::utils::DTypeToString(y_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
            output_default_dtype = y_dtype.value();
        }
        aclDataType output_acl_type = c10_npu::GetAclDataType(output_default_dtype);
        at::ScalarType output_scalar_type = npu_preparation::convert_to_scalar_type(output_acl_type);
        // 输出的output_tensor需要自己推导，all2allmm的output_size就是左矩阵x1的m和右矩阵x2的n
        auto output_size = {x1.size(0), x2.size(1)};
        // output_tensor按照实际的shape和dtype去创建
        at::Tensor output_tensor = npu_preparation::apply_tensor_without_format(output_size, c10::dtype(output_scalar_type));

        // attr
        char *group_ptr = const_cast<char *>(hcom.data());

        // 如果是海思自定义的dtype的话，那么这里就需要使用一个wrapper
        // 让aclnn接口识别到传入的tensor具体的dtype类型，相当于打一个标签，标记真实的属性
        // TensorWrapper是一个类型适配器，将PyTorch的at::Tensor包装成一个带有显示aclDataType的结构体，让底层ACL接口能正确识别张量的真实数据类型
        // x1.scalar_type()即PyTorch的ScalarType，c10_npu::GetAclDataType()和npu_preparation::convert_to_acl_data_type()都会返回一个aclDataType
        TensorWrapper x1_wrapper = {x1, (x1_dtype.has_value()) ? c10_npu::GetAclDataType(x1_dtype.value()) : npu_preparation::convert_to_acl_data_type(x1.scalar_type())}; 
        TensorWrapper x2_wrapper = {x2, (x2_dtype.has_value()) ? c10_npu::GetAclDataType(x2_dtype.value()) : npu_preparation::convert_to_acl_data_type(x2.scalar_type())};
        TensorWrapper x1_scale_wrapper = {x1_scale, (x1_scale.has_value()) ? c10_npu::GetAclDataType(x1_scale.value()) : npu_preparation::convert_to_acl_data_type(x1_scale.scalar_type())};
        TensorWrapper x2_scale_wrapper = {x2_scale, (x2_scale.has_value()) ? c10_npu::GetAclDataType(x2_scale.value()) : npu_preparation::convert_to_acl_data_type(x2_scale.scalar_type())};
        
        // 前面的wrapper打包传进去之后，这里直接调用aclnn接口
        // 根据x1_quant_mode和x2_quant_mode判断调用非量化还是量化aclnn接口
        if (x1_quant_mode == 0 && x2_quant_mode == 0) {
            EXEC_NPU_CMD(aclnnMatmulAlltoAllGetWorkspaceSize, x1_wrapper, x2_wrapper, bias, 
                all2all_axes_list, "test_matmul_all-to-all", transpose_x1, transpose_x2, output_tensor);
        }
        else {
            EXEC_NPU_CMD(aclnnQuantMatmulAlltoAllGetWorkspaceSize, x1_wrapper, x2_wrapper, bias, 
                x1_scale_wrapper, x2_scale_wrapper, common_scale, x1_offset, x2_offset, 
                "test_quant_matmul_all-to-all", all2all_axes_list, x1_quant_mode, x2_quant_mode, common_quant_mode, 
                comm_quant_dtype, group_sizes, transpose_x1, transpose_x2, output_tensor);
        }
        return output_tensor;
    }

} // namespace op_api
