// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
at::Tensor gen_mask_impl(at::Tensor & self, at::IntArrayRef size, double p)
{
    const int64_t BYTE_BIT = 8;
    const int64_t DATA_ALIGN = 128;
    int64_t numels = c10::multiply_integers(size);

    uint64_t length = (static_cast<uint64_t>(numels) + DATA_ALIGN - 1) / DATA_ALIGN * DATA_ALIGN / BYTE_BIT;
    c10::TensorOptions options = self.options();
    at::Tensor mask =
        npu_preparation::apply_tensor_without_format(at::IntArrayRef{length}, options.dtype(at::kByte));  //output mask unit8
    // at::Tensor y = npu_preparation::apply_tensor_without_format(at::IntArrayRef{1}, options.dtype(at::kByte));  //y , no use
    c10::SmallVector<int64_t, SIZE> shapeSize = {numels};
    at::IntArrayRef shapeArray = at::IntArrayRef(shapeSize);
    float p_cast = static_cast<float>(p); 
    // aclDataType probDataType = at_npu::native::OpPreparation::convert_to_acl_data_type(self.scalar_type());
    // std::cout<<"p_cast = "<<p_cast<<std::endl;
    EXEC_NPU_CMD(aclnnDropoutV3, shapeArray, self, p_cast, mask, self);
    return mask;
}
}  // namespace


::std::tuple<at::Tensor,at::Tensor> _npu_dropout_gen_mask_v2(at::Tensor & self, at::IntArrayRef size, double p, c10::optional<bool> parallel, c10::optional<bool> sync)
{
    TORCH_CHECK(p >= 0 && p <= 1, "dropout probability has to be between 0 and 1, but got ", p,
                OPS_ERROR(ErrCode::VALUE));
    at::Tensor mask;
    bool parallel_value = parallel.value_or(true);
    if (parallel_value) {
        auto original_stream = c10_npu::getCurrentNPUStream();
        {
            // During the life cycle of this raii instance, the calcu stream is set as the
            // secondary stream, and tasks are distributed to the secondary stream. At the
            // same time, according to the one-stream-one-pool principle, memory is also
            // alloced from the pool of the secondary stream.
            c10_npu::SecondaryStreamGuard guard(c10_npu::getCurrentSecondaryStream());
            mask = gen_mask_impl(self, size, p);
            bool sync_value = sync.value_or(false);
            if (sync_value) {
                NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(original_stream));
            }
        }
    } else {
        mask = gen_mask_impl(self, size, p);
    }
    return std::make_tuple(std::move(self), std::move(mask));
}
}  // namespace op_api
