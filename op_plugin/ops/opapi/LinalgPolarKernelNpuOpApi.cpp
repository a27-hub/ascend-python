// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// You may not use this file except in compliance with the License.
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
#if VERSION_BETWEEN(V2R14, VERSION_NEWEST)
namespace {
at::Tensor materialize_conj_view(const at::Tensor& tensor)
{
    return tensor.resolve_conj().contiguous();
}

// Input checks. The messages match the CPU/CUDA side (aten::native::checkIsMatrix,
// checkFloatingOrComplex and the row/column check of TORCH_META_FUNC(linalg_polar)) so that
// generic OpInfo tests can compare across backends.
void check_linalg_polar_inputs(const at::Tensor& A)
{
    constexpr int64_t kMatrixDim = 2;
    TORCH_CHECK(
        A.dim() >= kMatrixDim,
        "linalg.polar: The input tensor A must have at least 2 dimensions, but got ",
        A.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        A.size(-2) >= A.size(-1),
        "linalg.polar: input must have at least as many rows as columns, but got ",
        A.size(-2),
        " by ",
        A.size(-1),
        " matrices",
        OPS_ERROR(ErrCode::PARAM));

    const auto dtype = A.scalar_type();
    TORCH_CHECK(
        at::isFloatingType(dtype) || at::isComplexType(dtype),
        "linalg.polar: Expected a floating point or complex tensor as input. Got ",
        dtype,
        OPS_ERROR(ErrCode::TYPE));
    // NPU restriction: the adapted aclnnSvd supports float32/float64/complex64/complex128, while
    // aclnnMatmul only supports bfloat16/float16/float32 (no float64, no complex), so the two
    // operators intersect at float32 only.
    TORCH_CHECK(
        dtype == at::kFloat,
        "linalg.polar: only float32 is supported on the NPU backend, but got ",
        dtype,
        OPS_ERROR(ErrCode::TYPE));
}

// Polar decomposition A = U H from the reduced SVD A = Up diag(S) Vh:
//     U = Up Vh,   H = Vh^H diag(S) Vh
// H is Hermitianized explicitly to remove the round-off asymmetry of the reconstruction.
// The functional and the out variant share this computation.
std::tuple<at::Tensor, at::Tensor> linalg_polar_impl(const at::Tensor& A)
{
    const int64_t n = A.size(-1);
    if (A.numel() == 0) {
        auto u_sizes = A.sizes().vec();
        auto h_sizes = u_sizes;
        h_sizes[h_sizes.size() - 2] = n;
        return std::make_tuple(at::empty(u_sizes, A.options()), at::empty(h_sizes, A.options()));
    }

    constexpr int64_t kSvdMaxDim = 8;
    TORCH_CHECK(
        A.dim() <= kSvdMaxDim,
        "linalg.polar: aclnnSvd only supports 2 to 8 dimensions for non-empty input, but got ",
        A.dim(),
        OPS_ERROR(ErrCode::PARAM));

    auto svd = op_api::_linalg_svd(A, /*full_matrices=*/false, /*compute_uv=*/true, c10::nullopt);
    // aclnnSvd returns Vh as a conjugate view of a transposed allocation, and matmul does not
    // accept conjugate views on the NPU backend, so materialize the factors before using them.
    auto vh = materialize_conj_view(std::get<2>(svd));
    auto v = materialize_conj_view(vh.mH());
    auto h = v.matmul(std::get<1>(svd).unsqueeze(-1) * vh);
    h = 0.5 * (h + materialize_conj_view(h.mH()));
    return std::make_tuple(std::get<0>(svd).matmul(vh), std::move(h));
}
} // namespace

std::tuple<at::Tensor, at::Tensor> linalg_polar(const at::Tensor& A)
{
    check_linalg_polar_inputs(A);
    return linalg_polar_impl(A);
}

std::tuple<at::Tensor&, at::Tensor&> linalg_polar_out(const at::Tensor& A, at::Tensor& U, at::Tensor& H)
{
    check_linalg_polar_inputs(A);
    auto [u, h] = linalg_polar_impl(A);
    at_npu::native::OpPreparation::check_tensor({A}, U, u.scalar_type(), u.sizes());
    at_npu::native::OpPreparation::check_tensor({A}, H, h.scalar_type(), h.sizes());
    U.copy_(u);
    H.copy_(h);
    return std::tie(U, H);
}
#endif
} // namespace op_api
