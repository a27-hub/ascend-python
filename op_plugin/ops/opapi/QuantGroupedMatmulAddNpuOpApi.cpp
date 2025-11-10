// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

using npu_preparation = at_npu::native::OpPreparation;

const std::string TypeToString4Add(int64_t input_type)
{
    return c10_npu::IsCustomDType(input_type) ? c10_npu::CustomDataTypeToString(input_type)
                                              : c10::toString(static_cast<at::ScalarType>(input_type));
}

at::Tensor npu_add_quant_gmm_symint(const at::Tensor &self, const at::Tensor &x1, const at::Tensor &x2,
                                    const at::Tensor &x2_scale, const at::Tensor &group_list,
                                    const c10::optional<at::Tensor> &x1_scale, c10::optional<int64_t> group_list_type,
                                    c10::OptionalArrayRef<c10::SymInt> group_sizes, c10::optional<int64_t> x1_dtype,
                                    c10::optional<int64_t> x2_dtype, c10::optional<int64_t> x1_scale_dtype,
                                    c10::optional<int64_t> x2_scale_dtype)
// func: npu_add_quant_gmm_(Tensor(a!) self, Tensor x1, Tensor x2, Tensor x2_scale, Tensor group_list, *,
// Tensor? x1_scale=None, int? group_list_type=0, int[]? group_sizes=None, int? x1_dtype=None, int? x2_dtype=None,
// int? x1_scale_dtype=None, int? x2_scale_dtype=None) -> Tensor(a!) 对应的非原地npu实现
{
    TORCH_CHECK(c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910_95,
                "This operator currently only supports Ascend910_95 chips",
                OPS_ERROR(ErrCode::PARAM));
    // 确定能调到aclnn接口
    static const bool is_quant_grouped_matmul_inplace_add_available =
        check_aclnn_kernel_available("aclnnQuantGroupedMatmulInplaceAdd");
    TORCH_CHECK(is_quant_grouped_matmul_inplace_add_available,
                "Get aclnnQuantGroupedMatmulInplaceAdd or aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize faild, "
                "please upgrade CANN.",
                OPS_ERROR(ErrCode::PARAM));
    // x1_dtype、x2_dtype、x2_scale_dtype、x1_scale_dtype参数传入自定义数据类型的校验
    if (x1_dtype.has_value()) {
        TORCH_CHECK(x1_dtype.value() == static_cast<int64_t>(c10_npu::DType::HIFLOAT8),
                    "The optional parameter x1_dtype only supports hifloat8 or None, but now is ",
                    TypeToString4Add(x1_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
    }
    if (x2_dtype.has_value()) {
        TORCH_CHECK(x2_dtype.value() == static_cast<int64_t>(c10_npu::DType::HIFLOAT8),
                    "The optional parameter x2_dtype only supports hifloat8 or None, but now is ",
                    TypeToString4Add(x2_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
    }
    if (x2_scale_dtype.has_value()) {
        TORCH_CHECK(x2_scale_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E8M0),
                    "The optional parameter x2_scale_dtype only supports float8_e8m0 or None, but now is ",
                    TypeToString4Add(x2_scale_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
    }
    if (x1_scale_dtype.has_value()) {
        TORCH_CHECK(x1_scale_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E8M0),
                    "The optional parameter x1_scale_dtype only supports float8_e8m0 or None, but now is ",
                    TypeToString4Add(x1_scale_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
    }
    // check groupsize 为none
    TORCH_CHECK(!group_sizes.has_value(), "group_sizes is not supported currently. ", OPS_ERROR(ErrCode::VALUE));

    const at::Tensor &x1_scale_real = x1_scale.value_or(at::Tensor());
    int64_t group_list_type_value = group_list_type.value_or(0);
    int64_t group_size = 0;

    TensorWrapper x1_wrapper = {x1, x1_dtype.has_value() ? c10_npu::GetAclDataType(x1_dtype.value())
                                                         : npu_preparation::convert_to_acl_data_type(x1.scalar_type())};
    TensorWrapper x2_wrapper = {x2, x2_dtype.has_value() ? c10_npu::GetAclDataType(x2_dtype.value())
                                                         : npu_preparation::convert_to_acl_data_type(x2.scalar_type())};
    TensorWrapper x2_scale_wrapper = {
        x2_scale, x2_scale_dtype.has_value() ? c10_npu::GetAclDataType(x2_scale_dtype.value())
                                             : npu_preparation::convert_to_acl_data_type(x2_scale.scalar_type())};
    TensorWrapper x1_scale_wrapper = {
        x1_scale_real,
        x1_scale_dtype.has_value()
            ? c10_npu::GetAclDataType(x1_scale_dtype.value())
            : (x1_scale.has_value() ? npu_preparation::convert_to_acl_data_type(x1_scale_real.scalar_type())
                                    : aclDataType::ACL_FLOAT)};

    auto self_copy = self.clone();
    EXEC_NPU_CMD(aclnnQuantGroupedMatmulInplaceAdd, x1_wrapper, x2_wrapper, x1_scale_wrapper, x2_scale_wrapper,
                 group_list, self_copy, group_list_type_value, group_size);

    return self_copy;
}

}  // namespace op_api