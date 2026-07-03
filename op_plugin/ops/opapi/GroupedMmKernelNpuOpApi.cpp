// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include <ATen/Context.h>
#include <ATen/native/GroupedMMUtils.h>
#include <ATen/native/TensorFactories.h>

#include "op_plugin/OpApiInterface.h"

namespace op_api {
namespace {

bool should_fill_empty_deterministic()
{
    return at::globalContext().deterministicAlgorithms() &&
           at::globalContext().deterministicFillUninitializedMemory();
}

void fill_output_storage_deterministic_(at::Tensor &out)
{
    if (C10_UNLIKELY(!should_fill_empty_deterministic())) {
        return;
    }

    const at::Storage &storage = out.unsafeGetTensorImpl()->unsafe_storage();
    const int64_t storage_numel = storage.nbytes() / out.itemsize();
    if (storage_numel <= 0) {
        return;
    }

    at::Tensor storage_view = at::empty(
        {},
        at::TensorOptions().dtype(out.scalar_type()).device(out.device()));
    storage_view.set_(
        storage,
        /*storage_offset=*/0,
        /*size=*/{storage_numel},
        /*stride=*/{1});
    at::native::fill_empty_deterministic_(storage_view);
}

} // namespace

at::Tensor _grouped_mm(
    const at::Tensor &mat_a,
    const at::Tensor &mat_b,
    const c10::optional<at::Tensor> &offs,
    const c10::optional<at::Tensor> &bias,
    c10::optional<c10::ScalarType> out_dtype)
{
    at::native::_grouped_mm_validate_inputs(mat_a, mat_b, offs, bias, out_dtype);
    const auto out_dtype_ = at::native::_resolve_grouped_mm_out_dtype(mat_a, mat_b, out_dtype);
    at::Tensor out = at::native::create_grouped_gemm_output_tensor(mat_a, mat_b, offs, out_dtype_);
    fill_output_storage_deterministic_(out);
    at::native::_grouped_mm_fallback(mat_a, mat_b, offs, bias, out_dtype, out);
    return out;
}

} // namespace op_api
