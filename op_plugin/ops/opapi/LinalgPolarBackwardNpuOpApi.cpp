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

struct PolarEigen {
    at::Tensor q;   // eigenvectors of H
    at::Tensor qh;  // Q^H, materialized
    at::Tensor s;   // eigenvalues, non-negative
};

// H is Hermitian positive semi-definite, so its SVD coincides with its eigendecomposition:
//     H = Q diag(s) Q^H,  s >= 0
// Taking (Q, s) from the SVD of H replaces the at::linalg_eigh of the CPU/CUDA formula, which
// has no NPU kernel.
PolarEigen polar_eigh_via_svd(const at::Tensor& H)
{
    auto svd = op_api::_linalg_svd(H, /*full_matrices=*/false, /*compute_uv=*/true, c10::nullopt);
    PolarEigen eig;
    eig.q = std::get<0>(svd).contiguous();
    eig.qh = materialize_conj_view(eig.q.mH());
    eig.s = std::get<1>(svd);
    return eig;
}

// Solves H X + X H = R for Hermitian positive (semi-)definite H = Q diag(s) Q^H.
at::Tensor polar_sylvester_solve(const PolarEigen& eig, const at::Tensor& R)
{
    auto denom = eig.s.unsqueeze(-1) + eig.s.unsqueeze(-2);
    auto inner = at::matmul(at::matmul(eig.qh, R), eig.q).div(denom);
    auto out = at::matmul(at::matmul(eig.q, inner), eig.qh);
    return 0.5 * (out + materialize_conj_view(out.mH()));
}

// X H^{-1} reusing the eigendecomposition of H.
at::Tensor polar_apply_hinv(const PolarEigen& eig, const at::Tensor& X)
{
    return at::matmul(at::matmul(X, eig.q).div(eig.s.unsqueeze(-2)), eig.qh);
}
} // namespace

// Backward of torch.linalg.polar (VJP). Mirrors the CPU/CUDA linalg_polar_backward, with the
// eigendecomposition of H taken from the SVD:
//     C = U^H gU
//     gA += (gU - U C) H^{-1} + U syl(C - C^H)
//     gA += A syl(gH + gH^H)
// Rank-deficient A leaves 1 / (s_i + s_j) undefined, same as upstream.
at::Tensor npu_linalg_polar_backward(
    const c10::optional<at::Tensor>& grad_U,
    const c10::optional<at::Tensor>& grad_H,
    const at::Tensor& A,
    const at::Tensor& U,
    const at::Tensor& H)
{
    const bool has_grad_u = grad_U.has_value() && grad_U->defined();
    const bool has_grad_h = grad_H.has_value() && grad_H->defined();
    if (!has_grad_u && !has_grad_h) {
        return {};
    }

    const auto eig = polar_eigh_via_svd(H);
    at::Tensor grad_A;
    if (has_grad_u) {
        const auto& gU = *grad_U;
        auto c = at::matmul(materialize_conj_view(U.mH()), gU);
        // The part of gU outside range(U) only needs H^{-1}; the part inside goes through the
        // Sylvester solve.
        auto normal = polar_apply_hinv(eig, gU - at::matmul(U, c));
        grad_A = normal +
            at::matmul(U, polar_sylvester_solve(eig, c - materialize_conj_view(c.mH())));
    }
    if (has_grad_h) {
        const auto& gH = *grad_H;
        auto from_h =
            at::matmul(A, polar_sylvester_solve(eig, gH + materialize_conj_view(gH.mH())));
        grad_A = grad_A.defined() ? grad_A + from_h : std::move(from_h);
    }
    return grad_A;
}

// Forward-mode derivative (JVP) of torch.linalg.polar:
//     dH = syl(dA^H A + A^H dA),   dU = (dA - U dH) H^{-1}
std::tuple<at::Tensor, at::Tensor> npu_linalg_polar_jvp(
    const at::Tensor& A,
    const at::Tensor& dA,
    const at::Tensor& U,
    const at::Tensor& H)
{
    const auto eig = polar_eigh_via_svd(H);
    auto a_h = materialize_conj_view(A.mH());
    auto d_m = at::matmul(materialize_conj_view(dA.mH()), A) + at::matmul(a_h, dA);
    auto dH = polar_sylvester_solve(eig, d_m);
    auto dU = polar_apply_hinv(eig, dA - at::matmul(U, dH));
    return std::make_tuple(std::move(dU), std::move(dH));
}
#endif
} // namespace op_api
