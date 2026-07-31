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

#include <cmath>

#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

tensor_list npu_batch_nms(const at::Tensor &self, const at::Tensor &scores, double score_threshold,
                          double iou_threshold, int64_t max_size_per_class, int64_t max_total_size,
                          bool change_coordinate_frame, bool transpose_box)
{
    TORCH_CHECK(self.numel() > 0, "The input tensor [self] is an empty tensor.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(scores.numel() > 0, "The input tensor [scores] is an empty tensor.", OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(self.dtype() == scores.dtype(),
        "self and scores must have the same dtype, but got self dtype: ", self.dtype(),
        " and scores dtype: ", scores.dtype(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.dtype() == at::kHalf || self.dtype() == at::kFloat,
        "npu_batch_nms only supports fp16 (at::kHalf) and fp32 (at::kFloat), but got: ",
        self.dtype(), OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(self.device() == scores.device(),
        "self and scores must be on the same device, but got self device: ", self.device(),
        " and scores device: ", scores.device(), OPS_ERROR(ErrCode::PARAM));

    at::Tensor boxes_contig = at_npu::native::custom_ops::npu_format_cast(self.contiguous(), ACL_FORMAT_ND);
    at::Tensor scores_contig = at_npu::native::custom_ops::npu_format_cast(scores.contiguous(), ACL_FORMAT_ND);

    TORCH_CHECK(boxes_contig.dim() == 4,
        "self (boxes) must be rank 4, but got rank: ", boxes_contig.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(scores_contig.dim() == 3,
        "scores must be rank 3, but got rank: ", scores_contig.dim(), OPS_ERROR(ErrCode::PARAM));

    int64_t B = boxes_contig.size(0);
    int64_t C = scores_contig.size(2);
    int64_t N;
    int64_t q;
    if (transpose_box) {
        TORCH_CHECK(boxes_contig.size(2) == 4,
            "When transpose_box is true, self.size(2) must be 4, but got: ",
            boxes_contig.size(2), OPS_ERROR(ErrCode::PARAM));
        N = boxes_contig.size(3);
        q = boxes_contig.size(1);
    } else {
        TORCH_CHECK(boxes_contig.size(3) == 4,
            "When transpose_box is false, self.size(3) must be 4, but got: ",
            boxes_contig.size(3), OPS_ERROR(ErrCode::PARAM));
        N = boxes_contig.size(1);
        q = boxes_contig.size(2);
    }

    TORCH_CHECK(B == scores_contig.size(0),
        "Batch dimension mismatch: self.size(0)=", B, " but scores.size(0)=",
        scores_contig.size(0), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(N == scores_contig.size(1),
        "N (num boxes) dimension mismatch: boxes N=", N, " but scores.size(1)=",
        scores_contig.size(1), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(q == 1 || q == C,
        "q (num classes in boxes) must be 1 or equal to C (num classes in scores=", C,
        "), but got q=", q, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(B > 0 && N > 0 && q > 0 && C > 0,
        "All dimensions B, N, q, C must be positive, but got B=", B, ", N=", N,
        ", q=", q, ", C=", C, OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(iou_threshold >= 0.0 && iou_threshold <= 1.0,
        "iou_threshold must be in [0, 1], but got: ", iou_threshold, OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(max_size_per_class > 0,
        "max_size_per_class must be > 0, but got: ", max_size_per_class, OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(max_total_size > 0,
        "max_total_size must be > 0, but got: ", max_total_size, OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(std::isfinite(score_threshold),
        "score_threshold must be finite, but got: ", score_threshold, OPS_ERROR(ErrCode::VALUE));

    TORCH_CHECK(!change_coordinate_frame,
        "change_coordinate_frame=True is not supported by npu_batch_nms because it requires "
        "a clip_window input that this API does not provide.", OPS_ERROR(ErrCode::PARAM));

    at::Tensor nmsed_boxes = npu_preparation::apply_tensor_with_format({B, max_total_size, 4}, self.options(), ACL_FORMAT_ND);
    at::Tensor nmsed_scores = npu_preparation::apply_tensor_with_format({B, max_total_size}, self.options(), ACL_FORMAT_ND);
    at::Tensor nmsed_classes = npu_preparation::apply_tensor_with_format({B, max_total_size}, self.options(), ACL_FORMAT_ND);
    at::Tensor nmsed_num = npu_preparation::apply_tensor_with_format({B}, self.options().dtype(at::kInt), ACL_FORMAT_ND);

    float scoreThreshold = static_cast<float>(score_threshold);
    float iouThreshold = static_cast<float>(iou_threshold);
    at::Tensor clipWindow;
    at::Tensor numValidBoxes;
    EXEC_NPU_CMD(aclnnBatchMultiClassNonMaxSuppression,
        boxes_contig, scores_contig, clipWindow, numValidBoxes,
        scoreThreshold, iouThreshold,
        max_size_per_class, max_total_size,
        change_coordinate_frame, transpose_box,
        nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num);

    return std::tie(nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num);
}

} // namespace op_api
