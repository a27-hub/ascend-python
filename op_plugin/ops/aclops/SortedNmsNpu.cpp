// Copyright (c) 2023-2024 Huawei Technologies Co., Ltd
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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_sorted_nms(const at::Tensor &boxes, const at::Tensor &sorted_scores, const at::Tensor &input_indices, const at::Tensor &max_output_size, const at::Tensor &iou_thr, const at::Tensor &score_thr, int64_t offset)
{
    at::SmallVector<int64_t, SIZE> shape;
    shape.emplace_back(boxes.size(0));

    at::Tensor selected_indices = npu_preparation::apply_tensor(shape, boxes.options().dtype(at::kInt), boxes);

    at_npu::native::OpCommand cmd;
    cmd.Name("SortedNMS")
        .Input(boxes, "boxes")
        .Input(sorted_scores, "sorted_scores")
        .Input(input_indices, "input_indices")
        .Input(max_output_size, "max_output_size")
        .Input(iou_thr, "iou_threshold")
        .Input(score_thr, "score_threshold")
        .Output(selected_indices, "selected_indices")
        .Attr("offset", static_cast<int32_t>(offset))
        .Run();
    return selected_indices;
}
} // namespace acl_op