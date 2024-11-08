// Copyright (c) 2024 Huawei Technologies Co., Ltd
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
namespace {
const int64_t GATHER_B = 0;
const int64_t GATHER_S = 1;
const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;

void check_param_valid(const at::Tensor &x1, const at::Tensor &x2, const c10::optional<at::Tensor> &pertoken_scale,
    const at::ScalarType dtype, const int64_t gather_axis)
{
    TORCH_CHECK(x1.dim() == 3,
        "The dim num of input x1 should be 3, but got ", x1.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x2.dim() == 2,
        "The dim num of input x2 should be 2, but got ", x2.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((dtype == at::kHalf) || (dtype == at::kBFloat16) || (dtype == at::kChar),
        "Output dtype should be FLOAT16, INT8, BF16, but got ", dtype, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((gather_axis == GATHER_B) || (gather_axis == GATHER_S),
        "Gather axis should be 0: GATHER_B, 1: GATHER_S, but got ", gather_axis, OPS_ERROR(ErrCode::PARAM));

    if (pertoken_scale.has_value()) {
        const at::Tensor &pertoken_scale_real = pertoken_scale.value();
        TORCH_CHECK((pertoken_scale_real.dim() == 1) || (pertoken_scale_real.dim() == 2),
            "The pertoken_scale dim num should be 1 or 2, but got ", pertoken_scale_real.dim(),
            OPS_ERROR(ErrCode::PARAM));
    }
    return;
}
}
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_all_gather_quant_bmm(const at::Tensor &x1, const at::Tensor &x2,
    const at::Tensor &scale, c10::string_view hcom, int64_t world_size, const c10::optional<at::Tensor> &bias,
    const c10::optional<at::Tensor> &pertoken_scale, c10::optional<at::ScalarType> output_dtype,
    int64_t gather_axis, bool gather_output, int64_t comm_turn)
{
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    const at::Tensor &pertoken_scale_real = pertoken_scale.value_or(at::Tensor());
    at::ScalarType output_dtype_real = output_dtype.value_or(at::kHalf);
    char *group_ptr = const_cast<char*>(hcom.data());

    check_param_valid(x1, x2, pertoken_scale, output_dtype_real, gather_axis);

    int64_t b = x1.size(DIM_0);
    int64_t s = x1.size(DIM_1);
    int64_t h = x1.size(DIM_2);
    int64_t n = x2.size(DIM_1);

    if (gather_axis == GATHER_B) {
        b = b * world_size;
    } else { // GATHER_S
        s = s * world_size;
    }

    at::Tensor y_out = at::empty({b, s, n}, x1.options().dtype(output_dtype_real));

    at::Tensor gather_out_x1 {nullptr};
    at::Tensor gather_out_pertoken {nullptr};
    if (gather_output) {
        gather_out_x1 = at::empty({b, s, h}, x1.options().dtype(at::kChar));
    }

    // if has pertoken_scale, gather_pertoken_scale should output
    if (gather_output && pertoken_scale.has_value()) {
        if (pertoken_scale_real.dim() == 1) {
            gather_out_pertoken = at::empty({b * s}, x1.options().dtype(pertoken_scale_real.scalar_type()));
        } else { // case of dim = 2
            gather_out_pertoken = at::empty({b, s}, x1.options().dtype(pertoken_scale_real.scalar_type()));
        }
    }

    bool trans_x2 = false;

    EXEC_NPU_CMD(aclnnAllGatherQuantBatchMatMul, x1, x2, scale, bias_real, pertoken_scale_real,
                 trans_x2, group_ptr, world_size, gather_axis, comm_turn, y_out, gather_out_x1, gather_out_pertoken);
    return std::tie(y_out, gather_out_x1, gather_out_pertoken);
}

}
