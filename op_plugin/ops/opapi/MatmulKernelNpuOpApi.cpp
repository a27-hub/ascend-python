// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
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
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/KernelNpuOutputSize.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

const int8_t ALLOW_FP32_DOWN_PRECISION = 1;
const int8_t KEEP_DTYPE = 0;

static bool is_nz_format(const at::Tensor &mat2)
{   
    const torch_npu::NPUStorageDesc &tensor_desc = torch_npu::NPUBridge::GetNpuStorageImpl(mat2)->npu_desc_;
    return tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ;
}

static c10::SmallVector<int64_t, op_infer::SIZE> get_output_size(const at::Tensor &tensor1, const at::Tensor &tensor2)
{
    c10::SmallVector<int64_t, op_infer::SIZE> output_size;
    auto dim_tensor1 = tensor1.dim();
    auto dim_tensor2 = tensor2.dim();

    TORCH_CHECK(dim_tensor1 > 0 && dim_tensor2 > 0, "matmul got error dimentions: ", "(", dim_tensor1, ", ",
                dim_tensor2, ")", OPS_ERROR(ErrCode::PARAM));

    // aclnn weight NZ format only support 2D * 2D
    if (is_nz_format(tensor2) && !is_nz_format(tensor1)) {
        TORCH_CHECK(dim_tensor1 == 2 && dim_tensor2 == 2,
            "matmul with weight NZ got error dimentions: ",
            "(", dim_tensor1, ", ", dim_tensor2, ")", OPS_ERROR(ErrCode::PARAM));
    }            
    if (dim_tensor1 == 1 && dim_tensor2 == 1) {
        output_size = {};
    } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
        output_size = {tensor1.size(0)};
    } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
        output_size = {tensor2.size(1)};
    } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
        output_size = {tensor1.size(0), tensor2.size(1)};
    } else if (dim_tensor1 >= 3 && (dim_tensor2 == 1 || dim_tensor2 == 2)) {
        // t1:(N, n, m) * t2:(m, p)
        auto size1 = tensor1.sizes();
        auto tmp = c10::SmallVector<int64_t, op_infer::SIZE>{tensor2.size(0), 1};
        auto size2 = dim_tensor2 == 1 ? tmp : tensor2.sizes();
        output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
        if (dim_tensor2 > 1) {
            output_size.push_back(size2[dim_tensor2 - 1]);
        }
    } else if ((dim_tensor1 == 1 || dim_tensor1 == 2) && dim_tensor2 >= 3) {
        auto tmp = c10::SmallVector<int64_t, op_infer::SIZE>{1, tensor1.size(0)};
        auto size1 = dim_tensor1 == 1 ? tmp : tensor1.sizes();
        auto size2 = tensor2.sizes();
        output_size.insert(output_size.end(), size2.begin(), size2.end() - 2);
        if (dim_tensor1 > 1) {
            output_size.push_back(size1[0]);
        }
        output_size.push_back(size2[dim_tensor2 - 1]);
    } else if (dim_tensor1 >= 3 && dim_tensor2 >= 3) {
        // t1:(b1, n, m1) * t2:(x2, m2, p)
        int64_t n = tensor1.size(-2);
        at::IntArrayRef batch_tensor1(tensor1.sizes().data(), dim_tensor1 - 2);
        int64_t p = tensor2.size(-1);
        at::IntArrayRef batch_tensor2(tensor2.sizes().data(), dim_tensor2 - 2);
        std::vector<int64_t> expand_batch_portion = at::infer_size(batch_tensor1, batch_tensor2);
        c10::SmallVector<int64_t, op_infer::SIZE> output_expand_size(expand_batch_portion);
        output_expand_size.insert(output_expand_size.end(), {n, p});
        output_size = output_expand_size;
    } else {
        TORCH_CHECK(false,
            "matmul got error sizes: ",
            "(", dim_tensor1, ", ", dim_tensor2, ")",
            OPS_ERROR(ErrCode::PARAM));
    }

    return output_size;
}

static inline void matmul_implement_npu(at::Tensor &out, const at::Tensor &self, const at::Tensor &mat2)
{
    // allow dicrease precision
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    // matmul & matmul_out have processed the aclop scenario
    if (is_nz_format(mat2) && !is_nz_format(self)) {
        EXEC_NPU_CMD(aclnnMatmulWeightNz, self, mat2, out, cube_math_type);
    } else {
        EXEC_NPU_CMD(aclnnMatmul, self, mat2, out, cube_math_type);
    }
    FLOP_COUNT(FlopCounter::mm_flop, self, mat2);
    return;
}

at::Tensor matmul_forward(const at::Tensor &self, const at::Tensor &mat2)
{
    at::NoNamesGuard guard;
    auto output_size = get_output_size(self, mat2);
    auto out = at_npu::native::OpPreparation::apply_tensor_without_format(output_size, self.options());
    matmul_implement_npu(out, self, mat2);
    return out;
}

at::Tensor matmul(const at::Tensor &tensor1, const at::Tensor &tensor2)
{
    if (at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1) && 
        at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2)) {
          DO_COMPATIBILITY(aclnnMatmul, acl_op::matmul(tensor1, tensor2));  
    } else {
         if (is_nz_format(tensor2) && !is_nz_format(tensor1)) {
             DO_COMPATIBILITY(aclnnMatmulWeightNz, acl_op::matmul(tensor1, tensor2));   
         } else {
            // set default to aclop when soc version < 95
            // because internal_format_opapi changed to all_version
            if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910_95) {
                return acl_op::matmul(tensor1, tensor2);
            }
            // throw exception when sov version >= 95, because only support aclnn
            const torch_npu::NPUStorageDesc &tensor_desc1 = torch_npu::NPUBridge::GetNpuStorageImpl(tensor1)->npu_desc_;
            const torch_npu::NPUStorageDesc &tensor_desc2 = torch_npu::NPUBridge::GetNpuStorageImpl(tensor2)->npu_desc_;
            TORCH_CHECK(false,
                "matmul got not support format in current device: ",
                "(",
                tensor_desc1.npu_format_,
                ", ",
                tensor_desc2.npu_format_,
                ")",
                OPS_ERROR(ErrCode::PARAM));
         }
    }
    // aclnn
    auto maybe_outnames = at::namedinference::compute_matmul_outnames(tensor1, tensor2);
    auto result = matmul_forward(tensor1, tensor2);
    at::namedinference::propagate_names_if_nonempty(result, maybe_outnames);
    return result;
}

at::Tensor &matmul_out(const at::Tensor &tensor1, const at::Tensor &tensor2, at::Tensor &result)
{
    if (at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor1) && 
        at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor2)) {
        DO_COMPATIBILITY(aclnnMatmul, acl_op::matmul_out(tensor1, tensor2, result));
    } else {
        if (is_nz_format(tensor2) && !is_nz_format(tensor1)) {
            DO_COMPATIBILITY(aclnnMatmulWeightNz, acl_op::matmul_out(tensor1, tensor2, result));
        } else {
            // set default to aclop when soc version < 95
            // because internal_format_opapi changed to all_version
            if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910_95) {
                return acl_op::matmul_out(tensor1, tensor2, result);
            }
            // throw exception when sov version >= 95, because only support aclnn
            const torch_npu::NPUStorageDesc &tensor_desc1 = torch_npu::NPUBridge::GetNpuStorageImpl(tensor1)->npu_desc_;
            const torch_npu::NPUStorageDesc &tensor_desc2 = torch_npu::NPUBridge::GetNpuStorageImpl(tensor2)->npu_desc_;
            TORCH_CHECK(false,
                "matmul got not support format in current device: ",
                "(",
                tensor_desc1.npu_format_,
                ", ",
                tensor_desc2.npu_format_,
                ")",
                OPS_ERROR(ErrCode::PARAM));
        }
    }
    // aclnn
    auto maybe_outnames = at::namedinference::compute_matmul_outnames(tensor1, tensor2);
    // matmul_out don't support backward
    auto output_size = get_output_size(tensor1, tensor2);
    at_npu::native::OpPreparation::check_tensor({tensor1, tensor2}, result, tensor1, output_size);
    matmul_implement_npu(result, tensor1, tensor2);
    at::namedinference::propagate_names_if_nonempty(result, maybe_outnames);
    return result;
}

}
