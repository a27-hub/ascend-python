// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;

    at::Tensor npu_rms_norm_quant(const at::Tensor &x, const at::Tensor &gamma, const at::Tensor &beta,
                                  const at::Tensor &scale, const at::Tensor &offset, double epsilon,
                                  c10::optional<int64_t> dst_type)
    {
        int64_t dst_type_value = dst_type.has_value() ? dst_type.value() : static_cast<int>(at::ScalarType::Char);
        aclDataType y_acltype = c10_npu::GetAclDataType(dst_type_value);
        at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);

        auto output_size = x.sizes();
        at::Tensor y = npu_preparation::apply_tensor_without_format(output_size, c10::dtype(scalar_dtype));

        TensorWrapper y_wrapper = {y, y_acltype};
        EXEC_NPU_CMD(aclnnRmsNormQuant, x, gamma, beta, scale, offset, epsilon, y_wrapper);
        return y_wrapper;
    }
}