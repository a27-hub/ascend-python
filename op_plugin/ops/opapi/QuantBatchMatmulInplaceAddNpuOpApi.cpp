// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpUtils.h"

namespace op_api {
namespace {
constexpr size_t GROUP_SIZE_DIM = 3;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr int64_t GROUP_MAX = 65535;

using npu_preparation = at_npu::native::OpPreparation;

bool is_transpose_last_two_dims(const at::Tensor& tensor)
{
    if (tensor.dim() < 2 || tensor.dim() > 6) {
        return false;
    }

    const int64_t dim1 = tensor.dim() - 1;
    const int64_t dim2 = tensor.dim() - LAST_SECOND_DIM_INDEX;
    if (tensor.stride(dim2) == 1 && tensor.stride(dim1) == tensor.size(dim2)) {
        int64_t tmp_nxd = tensor.size(dim1) * tensor.size(dim2);
        for (int64_t batch_dim = tensor.dim() - 3; batch_dim >= 0; --batch_dim) {
            if (tensor.stride(batch_dim) != tmp_nxd) {
                return false;
            }
            tmp_nxd *= tensor.size(batch_dim);
        }
        if (tensor.size(dim1) == 1 && tensor.size(dim2) == 1) {
            return false;
        }
        return true;
    }
    return false;
}

int64_t check_and_get_groups(at::IntArrayRef group_size_list)
{
    if (group_size_list.empty()) {
        return 0;
    }

    TORCH_CHECK(group_size_list.size() == GROUP_SIZE_DIM,
                "When group_sizes is not empty, it only supports input with 3 elements, but got ",
                group_size_list.size(), OPS_ERROR(ErrCode::PARAM));

    const int64_t group_m = static_cast<int64_t>(group_size_list[0]);
    const int64_t group_n = static_cast<int64_t>(group_size_list[1]);
    const int64_t group_k = static_cast<int64_t>(group_size_list[2]);
    TORCH_CHECK(group_m >= 0 && group_m <= GROUP_MAX && group_n >= 0 && group_n <= GROUP_MAX &&
                    group_k >= 0 && group_k <= GROUP_MAX,
                "group param value must conform to range [0, 65535]", OPS_ERROR(ErrCode::VALUE));

    return static_cast<int64_t>((static_cast<uint64_t>(group_m) << 32) +
                                (static_cast<uint64_t>(group_n) << 16) +
                                static_cast<uint64_t>(group_k));
}
}  // namespace

at::Tensor &npu_add_quant_matmul_(at::Tensor &self, const at::Tensor &x1, const at::Tensor &x2,
                                  const at::Tensor &x2_scale, const c10::optional<at::Tensor> &x1_scale,
                                  c10::OptionalIntArrayRef group_sizes, c10::optional<int64_t> x1_dtype,
                                  c10::optional<int64_t> x2_dtype, c10::optional<int64_t> x1_scale_dtype,
                                  c10::optional<int64_t> x2_scale_dtype)
    // func: npu_add_quant_matmul_(Tensor(a!) self, Tensor x1, Tensor x2, Tensor x2_scale, *,
    // Tensor? x1_scale=None, int[]? group_sizes=None, int? x1_dtype=None, int? x2_dtype=None,
    // int? x1_scale_dtype=None, int? x2_scale_dtype=None) -> Tensor(a!)
    // op_api: all_version
    // 对应的原地npu实现
{
    static const bool is_quant_batch_matmul_inplace_add_available =
        check_aclnn_kernel_available("aclnnQuantBatchMatmulInplaceAdd");
    TORCH_CHECK(is_quant_batch_matmul_inplace_add_available,
                "Get aclnnQuantBatchMatmulInplaceAdd or aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize failed, "
                "please upgrade CANN to version 9.0.0 or higher.",
                OPS_ERROR(ErrCode::PARAM));

    at::IntArrayRef group_size_list = group_sizes.value_or(at::IntArrayRef{});
    const int64_t group_size = check_and_get_groups(group_size_list);
    const bool transpose1 = is_transpose_last_two_dims(x1);
    const bool transpose2 = is_transpose_last_two_dims(x2);
    const at::Tensor &x1_scale_real = x1_scale.value_or(at::Tensor());

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

    EXEC_NPU_CMD(aclnnQuantBatchMatmulInplaceAdd, x1_wrapper, x2_wrapper, x1_scale_wrapper, x2_scale_wrapper,
                 self, transpose1, transpose2, group_size);

    return self;
}

}  // namespace op_api