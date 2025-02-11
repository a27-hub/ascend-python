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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/custom_functions/atb/atb_common.h"
#include "op_plugin/utils/custom_functions/atb/op_param_cache.h"
#include <acl/acl.h>

using namespace std;
namespace op_api
{
    using RopeParam = atb::infer::RopeParam;

    static at::Tensor sequenceLength;
    static int64_t previousTokenCount = -1;

    // Global caches for cos and sin values
    static at::Tensor cosCacheNeox;
    static at::Tensor sinCacheNeox;
    static at::Tensor cosCache;
    static at::Tensor sinCache;

    void InitializeCosSinCache(const at::Tensor &cos_sin_cache)
    {
        // Split the input cache tensor into two halves
        auto cosSinChunks = cos_sin_cache.chunk(2, /*dim=*/-1);

        // Prepare cosine caches with different repeat strategies
        cosCache = cosSinChunks[0].repeat_interleave(2, 1);
        sinCache = cosSinChunks[1].repeat_interleave(2, 1);
        cosCacheNeox = cosSinChunks[0].repeat({1, 2});
        sinCacheNeox = cosSinChunks[1].repeat({1, 2});
    }

    void npu_rope(const at::Tensor & positions, at::Tensor & query, at::Tensor & key, int64_t head_size, const at::Tensor & cos_sin_cache, bool is_neox)
    {
        // Initialize cos and sin caches if not already initialized
        if (!cosCache.defined() || !sinCache.defined())
        {
            InitializeCosSinCache(cos_sin_cache);
        }

        // Flatten positions tensor and select corresponding cos_sin values
        at::Tensor flatPositions = positions.flatten();
        int32_t currentTokenCount = flatPositions.size(0);

        // Select appropriate cos and sin caches based on RoPE variant
        at::Tensor cos = is_neox ? cosCacheNeox.index_select(0, flatPositions)
                                    : cosCache.index_select(0, flatPositions);
        at::Tensor sin = is_neox ? sinCacheNeox.index_select(0, flatPositions)
                                    : sinCache.index_select(0, flatPositions);

        // // Check if positions tensor has changed
        // const void *currentPositionsPtr = positions.data_ptr();

        // if (previousPositionsPtr != currentPositionsPtr)
        // {
        //     previousPositionsPtr = currentPositionsPtr;

        //     // Select appropriate cos and sin caches based on RoPE variant
        //     cos = is_neox ? cosCacheNeox.index_select(0, flatPositions)
        //                   : cosCache.index_select(0, flatPositions);
        //     sin = is_neox ? sinCacheNeox.index_select(0, flatPositions)
        //                   : sinCache.index_select(0, flatPositions);
        // }

        // Update sequence length tensor if necessary
        if (!sequenceLength.defined() || previousTokenCount != currentTokenCount)
        {
            previousTokenCount = currentTokenCount;
            sequenceLength = at::tensor({currentTokenCount}, at::kInt).to(query.device());
        }

        // Prepare rope parameters
        RopeParam ropeparam;
        ropeparam.rotaryCoeff = is_neox ? 2 : head_size;

        // Setup launch parameters
        ParamSetter parametter;
        parametter.Input(query)
            .Input(key)
            .Input(cos)
            .Input(sin)
            .Input(sequenceLength)
            .Output(query)
            .Output(key);

        // Get operation from cache and run command
        OpParamCache<RopeParam> &ropeParamCache = OpParamCache<RopeParam>::getInstance();
        auto opRope = ropeParamCache.getOperation(ropeparam, "RopeOperation");
        RunAtbCmd(opRope, parametter, "RopeOperation");
    }

} // namespace