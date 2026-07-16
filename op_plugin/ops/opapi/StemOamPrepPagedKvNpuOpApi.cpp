// Copyright (c) 2026 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor&, at::Tensor&> stem_oam_prep_paged_kv(
    const at::Tensor &kCache, const at::Tensor &vCache, const at::Tensor &kvIndices,
    const at::Tensor &kvSeqLens, const at::Tensor &kScaleCache, const at::Tensor &vScale,
    float lambdaMag, int64_t cacheLayout, int64_t kvBlockSize, int64_t stemBlockSize,
    int64_t stemStride, at::Tensor &kFlat, at::Tensor &vBias)
{
    EXEC_NPU_CMD(aclnnStemOamPrepPagedKv, kCache, vCache, kvIndices, kvSeqLens, kScaleCache,
                 vScale, lambdaMag, cacheLayout, kvBlockSize, stemBlockSize, stemStride, kFlat,
                 vBias);
    return std::tuple<at::Tensor&, at::Tensor&>(kFlat, vBias);
}
}  // namespace op_api
