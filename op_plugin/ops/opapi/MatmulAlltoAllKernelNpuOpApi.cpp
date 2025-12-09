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

#include <set>
#include <vector>
#include "op_plugin/OpApiInterface.h"
#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpUtils.h"


namespace op_api {

using npu_preparation = at_npu::native::OpPreparation;

static const size_t GROUP_DIM_1 = 1;
static const size_t GROUP_DIM_2 = 2;
static const uint64_t GROUP_MAX = 65535UL;
static const int TWO_DIMS = 2;
static const int64_t NEG_ONE = -1;
static const int64_t NEG_TWO = -2;

// x valid dtype
const std::set<int64_t> SUPPORT_X_DTYPE_LIST {
    static_cast<int64_t>(c10_npu::DType::FLOAT16),
    static_cast<int64_t>(c10_npu::DType::BFLOAT16)
};

// output valid dtype
const std::set<int64_t> SUPPORT_Y_DTYPE_LIST {
    static_cast<int64_t>(c10_npu::DType::FLOAT16),
    static_cast<int64_t>(c10_npu::DType::BFLOAT16)
};

// 校验all2all_axes是否合法，在matmulalltoall中，该值只能为[-1, -2]或空，为空时会做兼容性处理。
at::IntArrayRef check_and_get_axes(at::IntArrayRef all2all_axes_list)
{
    at::IntArrayRef all2all_axes_res = {-1, -2};
    if (all2all_axes_list.empty()) {
        return all2all_axes_res;
    }
    size_t all2all_axes_dim = all2all_axes_list.size();
    TORCH_CHECK(all2all_axes_dim == TWO_DIMS, "all2all_axes_dim only support 2D, but got ",
                all2all_axes_dim, OPS_ERROR(ErrCode::PARAM));
    int64_t dim1 = static_cast<int64_t>(all2all_axes_list[0]);
    int64_t dim2 = static_cast<int64_t>(all2all_axes_list[1]);
    bool invalid_all2all_axes = ((dim1 == NEG_ONE) && (dim2 == NEG_TWO));
    TORCH_CHECK(invalid_all2all_axes, "In MatmulAlltoAll, all2all_axes value must be [-1, -2], but it is ", all2all_axes_list, OPS_ERROR(ErrCode::VALUE));
    return all2all_axes_list;
}

at::Tensor npu_matmul_all_to_all_symint(const at::Tensor &x1, const at::Tensor &x2, c10::string_view hcom,
                                        const c10::optional<at::Tensor>& bias, c10::OptionalArrayRef<c10::SymInt> all2all_axes)
{
    // 校验x的shape, 2维(m, k) or (k, n)
    TORCH_CHECK(x1.dim() == TWO_DIMS, "The x1 input of mm is required to be 2D, but the actual x1 input is ", x.dim(),
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x2.dim() == TWO_DIMS, "The x2 input of mm is required to be 2D, but the actual x2 input is ", x.dim(),
                OPS_ERROR(ErrCode::PARAM));

    // 校验all2all_axes，该值只能为空或者[-1,-2]
    at::IntArrayRef all2all_axes_list = {};
    if (all2all_axes.has_value())
    {
        all2all_axes_list = c10::asIntArrayRefUnchecked(all2all_axes.value());
    }
    auto all2all_axes_value = check_and_get_axes(all2all_axes_list);

    // pta主要是为了推导output的shape和dtype，如果这里的output_dtype没有传入，则默认是bf16
    int64_t output_default_dtype = static_cast<int64_t>(at::ScalarType::BFloat16);
    if (y_dtype.has_value())
    {
        // 校验output的dtype: float16/bfloat16
        TORCH_CHECK(SUPPORT_Y_DTYPE_LIST.find(y_dtype.value()) != SUPPORT_Y_DTYPE_LIST.end(),
                    "The optional parameter y_dtype only supports float16/bfloat16, but now is ",
                    op_plugin::utils::DTypeToString(y_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
        output_default_dtype = y_dtype.value();
    }
    aclDataType output_acl_type = c10_npu::GetAclDataType(output_default_dtype);
    at::ScalarType output_scalar_type = npu_preparation::convert_to_scalar_type(output_acl_type);
    // 推导输出output_tensor的shape
    // x1不允许转置，因此可以确定m和k的取值
    out_m = x1.size(0);
    dim_k = x1.size(1);
    out_n = x2.size(0) == dim_k ? x2.size(1) : x2.size(0);
    auto output_size = {out_m, out_n};
    // output_tensor按照实际的shape和dtype去创建
    at::Tensor output_tensor = npu_preparation::apply_tensor_without_format(output_size, c10::dtype(output_scalar_type));

    // 生成aclnn接口需要的默认值
    char *group_ptr = const_cast<char *>(hcom.data());
    bool transpose_x1 = false;
    bool transpose_x2 = false;

    // 调用aclnn接口
    EXEC_NPU_CMD(aclnnMatmulAlltoAllGetWorkspaceSize, x1, x2, bias, all2all_axes, group_ptr, transpose_x1, transpose_x2, output_tensor);
    return output_tensor;
}
} // namespace op_api