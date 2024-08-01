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

using npu_preparation = at_npu::native::OpPreparation;

at::TensorList create_tensor_list(const at::Tensor& tensor)
{
    std::vector<at::Tensor> tensorVector = {tensor};
    at::TensorList tensorList(tensorVector);
    return tensorList;
}

at::Tensor npu_gmm_tensor(const at::Tensor& x,
                          const at::Tensor& weight,
                          const at::Tensor& group_list,
                          const c10::optional<at::Tensor>& bias,
                          c10::optional<int64_t> group_type)
{
    at::TensorList tensorListX = create_tensor_list(x);
    at::TensorList tensorListWeight = create_tensor_list(weight);
    at::TensorList tensorListBias = bias.has_value() ? create_tensor_list(bias.value()) : at::TensorList();

    int64_t* groupListPtr = static_cast<int64_t *>(group_list.data_ptr());
    at::IntArrayRef groupListIntArrayRef(groupListPtr, group_list.numel());

    c10::TensorOptions options = x.options().dtype(x.scalar_type());
    auto dimNumWeight = weight.sizes().size();
    auto output_size = op_infer::array_to_small_vector({x.sizes()[0], weight.sizes()[dimNumWeight - 1]});
    std::vector<at::Tensor> y = {npu_preparation::apply_tensor_without_format(output_size, options)};
    at::TensorList result = at::TensorList(y);

    at::TensorList tensorListScale = at::TensorList();
    at::TensorList tensorListOffset = at::TensorList();
    at::TensorList tensorListAntiquantScale = at::TensorList();
    at::TensorList tensorListAntiquantOffset = at::TensorList();

    int64_t splitItem = 3;

    EXEC_NPU_CMD(aclnnGroupedMatmul, tensorListX, tensorListWeight, tensorListBias, tensorListScale, tensorListOffset,
                 tensorListAntiquantScale, tensorListAntiquantOffset, groupListIntArrayRef, splitItem, result);

    return y[0];
}
}  // namespace op_api

