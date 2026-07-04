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

#include "op_plugin/utils/Version.h"

#if VERSION_BETWEEN(V2R9, VERSION_NEWEST)

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/native/TensorFactories.h>

#include <algorithm>
#include <vector>

#include "op_plugin/OpApiInterface.h"

namespace op_api {
namespace {

bool check_valid_strides_and_return_transposed(const at::Tensor &mat)
{
    at::IntArrayRef tensor_strides = mat.strides();
    at::IntArrayRef tensor_sizes = mat.sizes();
    int64_t end_dim = mat.dim() - 1;
    int64_t alignment = 16 / mat.element_size();
    TORCH_CHECK(uint64_t(mat.data_ptr()) % 16 == 0, "expected data_ptr to be aligned to 16 bytes\n");
    if ((tensor_strides[end_dim - 1] == 1) &&
        (tensor_strides[end_dim] >= std::max<int64_t>(1, tensor_sizes[end_dim - 1]))) {
        TORCH_CHECK(tensor_strides[end_dim] % alignment == 0, "strides should be multiple of 16 bytes");
        return true;
    }
    if ((tensor_strides[end_dim] == 1) &&
        (tensor_strides[end_dim - 1] >= std::max<int64_t>(1, tensor_sizes[end_dim]))) {
        TORCH_CHECK(tensor_strides[end_dim - 1] % alignment == 0, "strides should be multiple of 16 bytes");
        return false;
    }
    TORCH_CHECK(false, "Invalid strides/sizes, got ", mat.strides(), " for strides and ", mat.sizes(), " for sizes");
    return false;
}

at::Tensor create_grouped_gemm_output_tensor(
    const at::Tensor &mat_a,
    const at::Tensor &mat_b,
    const c10::optional<at::Tensor> &offs,
    c10::ScalarType out_dtype)
{
    c10::SmallVector<int64_t, 3> out_size;
    const bool a_is_2d = mat_a.dim() == 2;
    const bool b_is_2d = mat_b.dim() == 2;
    if (a_is_2d) {
        if (b_is_2d) {
            out_size = {offs->size(0), mat_a.size(0), mat_b.size(1)};
        } else {
            TORCH_CHECK(offs->size(0) == mat_b.size(0), "matrix batch sizes have to match");
            out_size = {mat_a.size(0), mat_b.size(-1)};
        }
    } else {
        if (b_is_2d) {
            TORCH_CHECK(offs->size(0) == mat_a.size(0), "matrix batch sizes have to match");
            out_size = {mat_a.size(1), mat_b.size(1)};
        } else {
            TORCH_CHECK(mat_a.size(0) == mat_b.size(0), "batched dimension has to match");
            out_size = {mat_a.size(0), mat_a.size(1), mat_b.size(-1)};
        }
    }

    const auto last_dim = out_size.size() - 1;
    const auto alignment = 16 / c10::elementSize(out_dtype);
    const int64_t size_padded = (out_size[last_dim] + alignment - 1) / alignment * alignment;
    std::vector<int64_t> out_stride;
    if (a_is_2d != b_is_2d) {
        out_stride = {size_padded, 1};
    } else {
        out_stride = {out_size[1] * size_padded, size_padded, 1};
    }
    return at::empty_strided(out_size, out_stride, mat_a.options().dtype(out_dtype));
}

void grouped_mm_validate_inputs(
    const at::Tensor &mat_a,
    const at::Tensor &mat_b,
    const c10::optional<at::Tensor> &offs,
    const c10::optional<at::Tensor> &bias,
    c10::optional<c10::ScalarType> out_dtype)
{
    TORCH_CHECK((mat_a.dtype() == at::kBFloat16) || (mat_a.dtype() == at::kFloat) || (mat_a.dtype() == at::kHalf),
        "Expected mat_a to be Float32, BFloat16 or Float16 matrix, got ", mat_a.scalar_type());
    TORCH_CHECK((mat_b.dtype() == at::kBFloat16) || (mat_b.dtype() == at::kFloat) || (mat_b.dtype() == at::kHalf),
        "Expected mat_b to be Float32, BFloat16 or Float16 matrix, got ", mat_b.scalar_type());
    TORCH_CHECK(mat_a.dim() == 2 || mat_a.dim() == 3, "mat_a has to be 2 or 3d");
    TORCH_CHECK(mat_b.dim() == 2 || mat_b.dim() == 3, "mat_b has to be 2 or 3d");
    const bool a_is_2d = mat_a.dim() == 2;
    const bool b_is_2d = mat_b.dim() == 2;
    if (!a_is_2d || !b_is_2d) {
        TORCH_CHECK(mat_a.size(-1) == mat_b.size(-2),
            "contraction dimension of mat_a and mat_b must match");
    }

    check_valid_strides_and_return_transposed(mat_a);
    check_valid_strides_and_return_transposed(mat_b);
    TORCH_CHECK(offs.has_value() == (a_is_2d || b_is_2d),
        "Have to provide offsets if there is a 2d matrix, or no offset if both matrices are 3d");

    if (offs.has_value()) {
        TORCH_CHECK(offs->dim() == 1, "offs has to be 1D");
        TORCH_CHECK(offs->dtype() == at::kInt, "Offsets have to be int32");
    }
    TORCH_CHECK(!bias.has_value(), "Bias not supported yet");
}

c10::ScalarType resolve_grouped_mm_out_dtype(
    const at::Tensor &mat_a,
    [[maybe_unused]] const at::Tensor &mat_b,
    c10::optional<c10::ScalarType> out_dtype)
{
    const auto out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
    TORCH_CHECK(out_dtype_ == mat_a.dtype(), "Grouped gemm output dtype must match `mat_a` dtype");
    return out_dtype_;
}

void grouped_mm_fallback(
    const at::Tensor &mat_a,
    const at::Tensor &mat_b,
    const c10::optional<at::Tensor> &offs,
    [[maybe_unused]] const c10::optional<at::Tensor> &bias,
    [[maybe_unused]] c10::optional<c10::ScalarType> out_dtype,
    at::Tensor out)
{
    const bool a_is_2d = mat_a.dim() == 2;
    const bool b_is_2d = mat_b.dim() == 2;
    if (a_is_2d && !b_is_2d) {
        int64_t group_start_idx = 0;
        auto offs_cpu = offs.value().cpu();
        for (int64_t group_idx = 0; group_idx < offs_cpu.size(0); group_idx++) {
            int64_t group_end_idx = offs_cpu[group_idx].item<int>();
            auto mat_a_slice = mat_a.slice(0, group_start_idx, group_end_idx);
            auto out_slice = out.slice(0, group_start_idx, group_end_idx);
            at::mm_out(out_slice, mat_a_slice, mat_b[group_idx]);
            group_start_idx = group_end_idx;
        }
    } else if (!a_is_2d && b_is_2d) {
        int64_t group_start_idx = 0;
        auto offs_cpu = offs.value().cpu();
        for (int64_t group_idx = 0; group_idx < offs_cpu.size(0); group_idx++) {
            int64_t group_end_idx = offs_cpu[group_idx].item<int>();
            auto mat_b_slice = mat_b.slice(1, group_start_idx, group_end_idx);
            auto out_slice = out.slice(1, group_start_idx, group_end_idx);
            at::mm_out(out_slice, mat_a[group_idx], mat_b_slice);
            group_start_idx = group_end_idx;
        }
    } else if (a_is_2d && b_is_2d) {
        int64_t group_start_idx = 0;
        auto offs_cpu = offs.value().cpu();
        for (int64_t group_idx = 0; group_idx < offs_cpu.size(0); group_idx++) {
            int64_t group_end_idx = offs_cpu[group_idx].item<int>();
            auto mat_a_slice = mat_a.slice(1, group_start_idx, group_end_idx);
            auto mat_b_slice = mat_b.slice(0, group_start_idx, group_end_idx);
            auto out_slice = out[group_idx];
            at::mm_out(out_slice, mat_a_slice, mat_b_slice);
            group_start_idx = group_end_idx;
        }
    } else {
        at::bmm_out(out, mat_a, mat_b);
    }
}

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
    grouped_mm_validate_inputs(mat_a, mat_b, offs, bias, out_dtype);
    const auto out_dtype_ = resolve_grouped_mm_out_dtype(mat_a, mat_b, out_dtype);
    at::Tensor out = create_grouped_gemm_output_tensor(mat_a, mat_b, offs, out_dtype_);
    fill_output_storage_deterministic_(out);
    grouped_mm_fallback(mat_a, mat_b, offs, bias, out_dtype, out);
    return out;
}

} // namespace op_api

#endif
