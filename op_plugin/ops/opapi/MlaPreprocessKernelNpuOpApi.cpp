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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_mla_preprocess(
    const at::Tensor & input,
    const at::Tensor & gamma0,
    const at::Tensor & beta0,
    const at::Tensor & quantScale0,
    const at::Tensor & quantOffset0,
    const at::Tensor & wdqkv,
    const at::Tensor & deScale0,
    const at::Tensor & bias0,
    const at::Tensor & gamma1,
    const at::Tensor & beta1,
    const at::Tensor & quantScale1,
    const at::Tensor & quantOffset1,
    const at::Tensor & wuq,
    const at::Tensor & deScale1,
    const at::Tensor & bias1,
    const at::Tensor & gamma2,
    const at::Tensor & cos,
    const at::Tensor & sin,
    const at::Tensor & wuk,
    const at::Tensor & kvCache,
    const at::Tensor & kvCacheRope,
    const at::Tensor & slotmapping,
    const at::Tensor & ctkvScale,
    const at::Tensor & qNopeScale,
    int64_t N,
    int64_t headNum,
    int64_t cacheMode,
    int64_t quantMode)
{
    at::Tensor qOut0;
    at::Tensor kvCacheOut0;
    at::Tensor qOut1;
    at::Tensor kvCacheOut1;
    std::vector<int64_t> qOut0_shape;
    std::vector<int64_t> qOut1_shape;
    std::vector<int64_t> kvCacheOut1_shape;
    qOut0_shape.emplace_back(input.size(0));
    qOut0_shape.emplace_back(deScale1.size(0) / 192);
    qOut1_shape.emplace_back(input.size(0));
    qOut1_shape.emplace_back(deScale1.size(0) / 192);
    qOut1_shape.emplace_back(64);

    if(cacheMode == 0) {
        qOut0_shape.emplace_back(576);
        qOut0 = at::empty(qOut0_shape, at::TensorOptions().dtype(input.scalar_type()).device(input.options().device()));
        kvCacheOut0 = at::empty(kvCache.sizes(), at::TensorOptions().dtype(input.scalar_type()).device(input.options().device()));
    } else if(cacheMode == 1) {
        qOut0_shape.emplace_back(512);
        qOut0 = at::empty(qOut0_shape, at::TensorOptions().dtype(input.scalar_type()).device(input.options().device()));
        kvCacheOut0 = at::empty(kvCache.sizes(), at::TensorOptions().dtype(input.scalar_type()).device(input.options().device()));
        qOut1 = at::empty(qOut1_shape, at::TensorOptions().dtype(input.scalar_type()).device(input.options().device()));
        kvCacheOut1 = at::empty(kvCacheRope.sizes(), at::TensorOptions().dtype(input.scalar_type()).device(input.options().device()));
    } else if(cacheMode == 2) {
        qOut0_shape.emplace_back(512);
        qOut0 = at::empty(qOut0_shape, at::TensorOptions().dtype(at::ScalarType::Char).device(input.options().device()));
        kvCacheOut0 = at::empty(kvCache.sizes(), at::TensorOptions().dtype(at::ScalarType::Char).device(input.options().device()));
        qOut1 = at::empty(qOut1_shape, at::TensorOptions().dtype(input.scalar_type()).device(input.options().device()));
        kvCacheOut1 = at::empty(kvCacheRope.sizes(), at::TensorOptions().dtype(input.scalar_type()).device(input.options().device()));
    } else {
        qOut0_shape.emplace_back(512);
        qOut0 = at::empty(qOut0_shape, at::TensorOptions().dtype(input.scalar_type()).device(input.options().device()));
        kvCacheOut0 = at::empty(kvCache.sizes(), at::TensorOptions().dtype(input.scalar_type()).device(input.options().device()));
        qOut1 = at::empty(qOut1_shape, at::TensorOptions().dtype(input.scalar_type()).device(input.options().device()));
        kvCacheOut1 = at::empty(kvCacheRope.sizes(), at::TensorOptions().dtype(input.scalar_type()).device(input.options().device()));
    }

    EXEC_NPU_CMD(aclnnMlaPreprocess, input, gamma0, beta0, quantScale0, quantOffset0, wdqkv, deScale0,
                    bias0, gamma1, beta1, quantScale1, quantOffset1, wuq, deScale1,
                    bias1, gamma2, cos, sin, wuk, kvCache, kvCacheRope, slotmapping, ctkvScale,
                    qNopeScale, N, headNum, cacheMode, quantMode, qOut0, kvCacheOut0, qOut1, kvCacheOut1);

    return std::make_tuple(qOut0, kvCacheOut0, qOut1, kvCacheOut1);
}
}