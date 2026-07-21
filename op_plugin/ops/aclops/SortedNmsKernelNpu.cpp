// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/library.h>

#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor sorted_nms_meta(const at::Tensor &boxes, const at::Tensor &sorted_scores, const at::Tensor &input_indices,
    const at::Tensor &max_output_size, const at::Tensor &iou_threshold, const at::Tensor &score_threshold,
    int64_t offset) {
    (void)sorted_scores;
    (void)input_indices;
    (void)max_output_size;
    (void)iou_threshold;
    (void)score_threshold;
    (void)offset;
    // SortedNMS fills unselected positions with -1, so its storage is sized
    // for every candidate box even though the number of valid indices is dynamic.
    return at::empty({boxes.size(0)}, boxes.options().dtype(at::kInt));
}

at::Tensor sorted_nms(const at::Tensor &boxes, const at::Tensor &sorted_scores, const at::Tensor &input_indices,
    const at::Tensor &max_output_size, const at::Tensor &iou_threshold, const at::Tensor &score_threshold,
    int64_t offset) {
    TORCH_CHECK(boxes.dim() == 2 && boxes.size(1) == 4, "boxes must have shape [N, 4]", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(offset == 0 || offset == 1, "offset must be 0 or 1", OPS_ERROR(ErrCode::PARAM));

    at::Tensor selected_indices =
        npu_preparation::apply_tensor({boxes.size(0)}, boxes.options().dtype(at::kInt), boxes);
    at_npu::native::OpCommand cmd;
    cmd.Name("SortedNMS")
        .Input(boxes)
        .Input(sorted_scores)
        .Input(input_indices)
        .Input(max_output_size)
        .Input(iou_threshold)
        .Input(score_threshold)
        .Output(selected_indices)
        .Attr("offset", offset)
        .Run();
    return selected_indices;
}
} // namespace acl_op

namespace {
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("sorted_nms(Tensor boxes, Tensor sorted_scores, Tensor input_indices, "
          "Tensor max_output_size, Tensor iou_threshold, Tensor score_threshold, "
          "int offset=0) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, Meta, m) { m.impl("sorted_nms", TORCH_FN(acl_op::sorted_nms_meta)); }

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) { m.impl("sorted_nms", TORCH_FN(acl_op::sorted_nms)); }
} // namespace
