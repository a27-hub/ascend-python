// Copyright (c) 2025 Huawei Technologies Co., Ltd
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
#include <acl/acl.h>

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/custom_functions/atb/AtbCommon.h"

using namespace std;

namespace atb {
using PagedCacheLoadParam = atb::infer::PagedCacheLoadParam;
void _npu_paged_cache_load(const at::Tensor &key_cache_tensor, const at::Tensor &value_cache_tensor, const at::Tensor &block_table_tensor, const at::Tensor &context_lens, const at::Tensor &seq_starts, at::Tensor &key, at::Tensor &value)
{
    const c10::OptionalDeviceGuard device_guard(device_of(key_cache_tensor));
    OpParamCache<PagedCacheLoadParam>& pagedCacheLoadParamCache = OpParamCache<PagedCacheLoadParam>::getInstance();
    PagedCacheLoadParam pagedparam;
    pagedparam.kvCacheType = PagedCacheLoadParam::PAGED_CACHE_LOAD_ND;
    pagedparam.cuSeqLens = false;
    pagedparam.hasSeqStarts = true;

    ParamSetter paramsetter;
    paramsetter.Input(key_cache_tensor)
        .Input(value_cache_tensor)
        .Input(block_table_tensor)
        .Input(context_lens)
        .Input(key)
        .Input(value)
        .Input(seq_starts)
        .Output(key)
        .Output(value);
    auto opPaged = pagedCacheLoadParamCache.getOperation(pagedparam, "PagedCacheLoadOperation");
    RunAtbCmd(opPaged, paramsetter, "PagedCacheLoadOperation");

    return;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_paged_cache_load(Tensor key_cache_tensor, Tensor value_cache_tensor, Tensor block_table_tensor, Tensor context_lens, Tensor seq_starts, Tensor(a!) out_key_tensor, Tensor(b!) out_value_tensor) -> ()");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_paged_cache_load", TORCH_FN(atb::_npu_paged_cache_load));
}
}

}