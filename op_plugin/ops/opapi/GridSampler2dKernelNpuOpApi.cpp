// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

namespace op_api {
static const int64_t SLICE_WINDOW_TEP = 1;
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor grid_sampler_2d(const at::Tensor& self, const at::Tensor& grid, int64_t interpolation_mode,
                           int64_t padding_mode, bool align_corners) {
    auto aclnnGridSampleFuncAddr = GetOpApiFuncAddr("aclnnGridSample");
    auto aclnnGridSampleGetWorkspaceSizeFuncAddr = GetOpApiFuncAddr("aclnnGridSampleGetWorkspaceSize");
    if (aclnnGridSampleGetWorkspaceSizeFuncAddr == nullptr || aclnnGridSampleFuncAddr == nullptr) {
        DO_COMPATIBILITY(aclnnGridSampler2D, acl_op::grid_sampler_2d(self, grid, interpolation_mode, padding_mode,
                                                                     align_corners));
        auto outputSize = {self.size(0), self.size(1), grid.size(1), grid.size(2)};
        at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, self.options());
        EXEC_NPU_CMD(aclnnGridSampler2D, self, grid, interpolation_mode, padding_mode, align_corners, result);
        return result;
    } else {
        bool channelLast = false;
        int64_t schedulerMode = SLICE_WINDOW_TEP; // sliding window mode
        DO_COMPATIBILITY(aclnnGridSample, acl_op::grid_sampler_2d(self, grid, interpolation_mode, padding_mode,
                                                                  align_corners));
        auto outputSize = {self.size(0), self.size(1), grid.size(1), grid.size(2)};
        at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, self.options());
        EXEC_NPU_CMD(aclnnGridSample, self, grid, interpolation_mode, padding_mode, align_corners, channelLast, schedulerMode, result);
        return result;
    }
}
}
