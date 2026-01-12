// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpUtils.h"

namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;

    at::Tensor npu_confusion_transpose(
        const at::Tensor &self,
        at::IntArrayRef perm,
        at::IntArrayRef shape,
        bool transpose_first)
    {
        if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910_95) {
            return acl_op::npu_confusion_transpose(self, perm, shape, transpose_first);
        }
        DO_COMPATIBILITY(aclnnConfusionTranspose, acl_op::npu_confusion_transpose(self, perm, shape, transpose_first));
        c10::SmallVector<int64_t, SIZE> output_size;
        if (transpose_first) {
            output_size = op_infer::array_to_small_vector(shape);
        } else {
            auto shape_size = shape.size();
            for (uint i = 0; i < perm.size(); i++) {
                TORCH_CHECK(shape_size > perm[i], "npu_confusion_transpose input invalid, "
                                                  "shape has size ",
                            shape_size, " but perm[i] is, ", perm[i],
                            OPS_ERROR(ErrCode::PARAM));
                output_size.emplace_back(shape[perm[i]]);
            }
        }
        at::Tensor y = npu_preparation::apply_tensor_without_format(output_size, self.options());
        if (op_plugin::utils::is_nz_format(self)) {
            y = at_npu::native::custom_ops::npu_format_cast(y, ACL_FORMAT_FRACTAL_NZ);
        }

        EXEC_NPU_CMD(aclnnConfusionTranspose, self, perm, shape, transpose_first, y);
        return y;
    }

}
