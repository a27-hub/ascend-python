// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
#include <vector>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"

namespace op_api {
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr int64_t INT4_NUMS_IN_INT32 = 8;
using npu_preparation = at_npu::native::OpPreparation;

// bool static is_transpose_last_two_dims(const at::Tensor &tensor)
// {
//     if (tensor.dim() < 2 || tensor.dim() > 6) {
//         return false;
//     }
//     int64_t dim1 = tensor.dim() - 1;
//     int64_t dim2 = tensor.dim() - 2;
//     if (tensor.stride(dim2) == 1 && tensor.stride(dim1) == tensor.size(dim2)) {
//         int64_t tmpNxD = tensor.size(dim1) * tensor.size(dim2);
//         for (int64_t batchDim = tensor.dim() - 3; batchDim >= 0; batchDim--) {
//             if (tensor.stride(batchDim) != tmpNxD) {
//                 return false;
//             }
//             tmpNxD *= tensor.size(batchDim);
//         }
//         if (tensor.size(dim1) == 1 && tensor.size(dim2) == 1) {
//             return false;
//         }
//         return true;
//     }
//     return false;
// }

// uint64_t infer_out_batch_shape(const at::Tensor &x1, const at::Tensor &x2, std::vector<uint64_t> &batch_record)
{
    TORCH_CHECK(at_npu::native::FormatHelper::IsBaseFormatType(x2),
                "x2 should be in the original image format, but it is ",
                npu_preparation::get_tensor_npu_format(x2), OPS_ERROR(ErrCode::PARAM));
    uint64_t batch_val = 1;
    auto x1_dim_num = x1.dim();
    auto x2_dim_num = x2.dim();
    auto out_dim_num = std::max(x1_dim_num, x2_dim_num);
    auto &shape_long = x1_dim_num > x2_dim_num ? x1 : x2;
    auto &shape_short = x1_dim_num > x2_dim_num ? x2 : x1;
    int64_t vaild_offset = out_dim_num - std::min(x1_dim_num, x2_dim_num);
    for (int64_t i = 0; i < out_dim_num - LAST_SECOND_DIM_INDEX; i++) {
        auto short_dim = i < vaild_offset ? 1 : shape_short.size(i - vaild_offset);
        auto long_dim = shape_long.size(i);
        TORCH_CHECK(!(short_dim > 1 && long_dim > 1 && short_dim != long_dim),
                    "the x1 shape and x2 shape not supported for broadcast, the short_dim is ",
                    short_dim, " and  the long_dim is ", long_dim);
        uint64_t cur_batch_value = static_cast<uint64_t>(std::max(short_dim, long_dim));
        batch_val = batch_val * cur_batch_value;
        batch_record.push_back(cur_batch_value);
    }
    return batch_val;
}

#if VERSION_BETWEEN(V1R11, V1R11) || VERSION_BETWEEN(V2R1, VERSION_NEWEST)
at::Tensor npu_matmul_v5(const at::Tensor& x1, const at::Tensor& x2)
{
    c10::TensorOptions options;
    options = x1.options().dtype(at::kHalf);
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);

    EXEC_NPU_CMD(aclnnMatMulV5, x1, x2, result);

    return result;
}
#endif

}