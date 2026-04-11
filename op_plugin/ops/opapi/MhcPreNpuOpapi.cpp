// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/utils/OpUtils.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace {
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t BSND_DIMS = 4;
constexpr int64_t TND_DIMS = 3;
constexpr int64_t REMOVE_ONE_DIM = 1;
constexpr int64_t REMOVE_TWO_DIMS = 2;
constexpr int64_t ALPHA_NUMEL = 3;
constexpr int64_t NUM_THREE = 3;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_ONE = 1;

// aclnnMhcPre 依赖 CANN 9.0.0 及以上版本，此次校验避免旧版本环境出现找不到算子的兼容性问题。
inline void check_mhc_pre_supported()
{
    static const bool is_cann_ready = op_plugin::utils::is_gte_cann_version_900();
    static const bool is_aclnn_kernel_available = check_aclnn_kernel_available("aclnnMhcPre");
    TORCH_CHECK(
        is_cann_ready && is_aclnn_kernel_available,
        "torch_npu.npu_mhc_pre requires CANN >= 9.0.0 and aclnnMhcPre support. "
        "Please upgrade CANN.",
        OPS_ERROR(ErrCode::NOT_SUPPORT));
}

inline void check_mhc_pre_backward_supported()
{
    static const bool is_cann_ready = op_plugin::utils::is_gte_cann_version_900();
    static const bool is_aclnn_kernel_available = check_aclnn_kernel_available("aclnnMhcPre") && check_aclnn_kernel_available("aclnnMhcPreBackward");
    TORCH_CHECK(
        is_cann_ready && is_aclnn_kernel_available,
        "torch_npu.npu_mhc_pre_backward requires CANN >= 9.0.0, aclnnMhcPre and aclnnMhcPreBackward support. "
        "Please upgrade CANN.",
        OPS_ERROR(ErrCode::NOT_SUPPORT));
}

/**
 * @brief 构造 aclnnMhcPre 所需的输出张量。
 *
 * 该函数根据输入张量 x 的维度布局，并结合 phi 的第 0 维大小，
 * 预先创建算子执行所需的各个输出张量。
 * 当前支持两种 x 输入形式：
 * (1) 4 维输入：x 的形状为 [B, S, N, D]
 * (2) 3 维输入：x 的形状为 [T, N, D]
 *
 * @param x:   输入张量，要求为 3 维或 4 维。
 * @param phi: 输入张量，其第 0 维大小用于确定 outHmix 的最后一维。
 * @return 返回 6 个输出张量：
 * (outHin, outHpost, outHres, outInvRms, outHmix, outHpre)
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> construct_mhc_pre_outputs(
    const at::Tensor &x, const at::Tensor &phi)
{
    c10::TensorOptions hInOptions = x.options().dtype(x.dtype());
    c10::TensorOptions hOptions = x.options().dtype(at::kFloat);
    auto matK = phi.size(DIM_0);

    at::Tensor outHin;
    at::Tensor outHpost;
    at::Tensor outHres;
    at::Tensor outInvRms;
    at::Tensor outHmix;
    at::Tensor outHpre;

    if (x.dim() == BSND_DIMS) {
        auto batch = x.size(DIM_0);
        auto sequence = x.size(DIM_1);
        auto numResidual = x.size(DIM_2);
        auto dim = x.size(DIM_3);

        // 4 维输入场景：x 的 shape 为 [B, S, N, D]
        // 各输出张量 shape 说明如下：
        // - outHin    : [B, S, D]
        // - outHpost  : [B, S, N]
        // - outHres   : [B, S, N, N]
        // - outInvRms : [B, S]
        // - outHmix   : [B, S, K]，其中 K = phi.size(0)
        // - outHpre   : [B, S, N]
        c10::SmallVector<int64_t, BSND_DIMS - REMOVE_ONE_DIM> outHinSize;
        c10::SmallVector<int64_t, BSND_DIMS - REMOVE_ONE_DIM> outHpostSize;
        c10::SmallVector<int64_t, BSND_DIMS> outHresSize;
        c10::SmallVector<int64_t, BSND_DIMS - REMOVE_TWO_DIMS> outInvRmsSize;
        c10::SmallVector<int64_t, BSND_DIMS - REMOVE_ONE_DIM> outHmixSize;
        c10::SmallVector<int64_t, BSND_DIMS - REMOVE_ONE_DIM> outHpreSize;

        outHinSize.push_back(batch);
        outHinSize.push_back(sequence);
        outHinSize.push_back(dim);

        outHpostSize.push_back(batch);
        outHpostSize.push_back(sequence);
        outHpostSize.push_back(numResidual);

        outHresSize.push_back(batch);
        outHresSize.push_back(sequence);
        outHresSize.push_back(numResidual);
        outHresSize.push_back(numResidual);

        outInvRmsSize.push_back(batch);
        outInvRmsSize.push_back(sequence);

        outHmixSize.push_back(batch);
        outHmixSize.push_back(sequence);
        outHmixSize.push_back(matK);

        outHpreSize.push_back(batch);
        outHpreSize.push_back(sequence);
        outHpreSize.push_back(numResidual);

        outHin = at::empty(outHinSize, hInOptions);
        outHpost = at::empty(outHpostSize, hOptions);
        outHres = at::empty(outHresSize, hOptions);
        if (out_flag == 1) {
            outInvRms = at::empty(outInvRmsSize, hOptions);
            outHmix = at::empty(outHmixSize, hOptions);
            outHpre = at::empty(outHpreSize, hOptions);
        } else {
            // 空tensor
            outInvRms = at::empty({0}, hOptions);
            outHmix = at::empty({0}, hOptions);
            outHpre = at::empty({0}, hOptions);
        }
    } else {
        auto t = x.size(DIM_0);
        auto numResidual = x.size(DIM_1);
        auto dim = x.size(DIM_2);

        // 3 维输入场景：x 的 shape 为 [T, N, D]
        // 各输出张量 shape 说明如下：
        // - outHin    : [T, D]
        // - outHpost  : [T, N]
        // - outHres   : [T, N, N]
        // - outInvRms : [T]
        // - outHmix   : [T, K]，其中 K = phi.size(0)
        // - outHpre   : [T, N]
        c10::SmallVector<int64_t, TND_DIMS - REMOVE_ONE_DIM> outHinSize;
        c10::SmallVector<int64_t, TND_DIMS - REMOVE_ONE_DIM> outHpostSize;
        c10::SmallVector<int64_t, TND_DIMS> outHresSize;
        c10::SmallVector<int64_t, TND_DIMS - REMOVE_TWO_DIMS> outInvRmsSize;
        c10::SmallVector<int64_t, TND_DIMS - REMOVE_ONE_DIM> outHmixSize;
        c10::SmallVector<int64_t, TND_DIMS - REMOVE_ONE_DIM> outHpreSize;

        outHinSize.push_back(t);
        outHinSize.push_back(dim);

        outHpostSize.push_back(t);
        outHpostSize.push_back(numResidual);

        outHresSize.push_back(t);
        outHresSize.push_back(numResidual);
        outHresSize.push_back(numResidual);

        outInvRmsSize.push_back(t);

        outHmixSize.push_back(t);
        outHmixSize.push_back(matK);

        outHpreSize.push_back(t);
        outHpreSize.push_back(numResidual);

        outHin = at::empty(outHinSize, hInOptions);
        outHpost = at::empty(outHpostSize, hOptions);
        outHres = at::empty(outHresSize, hOptions);
        if (out_flag == 1) {
            outInvRms = at::empty(outInvRmsSize, hOptions);
            outHmix = at::empty(outHmixSize, hOptions);
            outHpre = at::empty(outHpreSize, hOptions);
        } else {
            // 空tensor
            outInvRms = at::empty({0}, hOptions);
            outHmix = at::empty({0}, hOptions);
            outHpre = at::empty({0}, hOptions);
        }
    }

    return std::make_tuple(outHin, outHpost, outHres, outInvRms, outHmix, outHpre);
}
} // namespace

namespace op_api {
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_mhc_pre(
    const at::Tensor &x, const at::Tensor &phi, const at::Tensor &alpha, const at::Tensor &bias,
    const c10::optional<at::Tensor> &gamma, double norm_eps, double hc_eps, int64_t out_flag)
{
    TORCH_CHECK(x.numel() > 0, "Input x should not be empty.");
    TORCH_CHECK(phi.numel() > 0, "Input phi should not be empty.");
    TORCH_CHECK(alpha.numel() == ALPHA_NUMEL, "Input alpha must have 3 elements, but got ", alpha.numel(), ".");
    TORCH_CHECK(bias.numel() > 0, "Input bias should not be empty.");

    TORCH_CHECK(x.dim() == TND_DIMS || x.dim() == BSND_DIMS, "Input x must be 3D or 4D, but got ", x.dim(), "D.");

    // out_flag 用于控制 aclnnMhcPre 是否实际写出全部输出：
    // - out_flag == 1：写出全部 6 个结果
    //   (outHin, outHpost, outHres, outInvRms, outHmix, outHpre)
    // - out_flag == 0：仅需要 outHin / outHpost / outHres，
    //   后 3 个输出在调用 aclnnMhcPre 时使用 nullTensor 占位，不参与实际写出
    TORCH_CHECK(out_flag == 0 || out_flag == 1, "out_flag must be 0 or 1, but got ", out_flag, ".");

    check_mhc_pre_supported();

    auto mhcPreOutput = construct_mhc_pre_outputs(x, phi);

    at::Tensor outHin = std::get<0>(mhcPreOutput);
    at::Tensor outHpost = std::get<1>(mhcPreOutput);
    at::Tensor outHres = std::get<2>(mhcPreOutput);
    at::Tensor outInvRms = std::get<3>(mhcPreOutput);
    at::Tensor outHmix = std::get<4>(mhcPreOutput);
    at::Tensor outHpre = std::get<5>(mhcPreOutput);

    at::Tensor nullTensor;
    if (out_flag == 1) {
        EXEC_NPU_CMD(aclnnMhcPre, x, phi, alpha, bias, gamma, norm_eps, hc_eps, outHin, outHpost,
                     outHres, outInvRms, outHmix, outHpre);
    } else {
        EXEC_NPU_CMD(aclnnMhcPre, x, phi, alpha, bias, gamma, norm_eps, hc_eps, outHin, outHpost,
                     outHres, nullTensor, nullTensor, nullTensor);
    }

    return std::make_tuple(outHin, outHpost, outHres, outInvRms, outHmix, outHpre);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_mhc_pre_backward(
        const at::Tensor &x, const at::Tensor &phi, const at::Tensor &alpha, 
        const at::Tensor &grad_h_in, const at::Tensor &grad_h_post, const at::Tensor &grad_h_res,
        const at::Tensor &inv_rms, const at::Tensor &h_mix, const at::Tensor &h_pre, const at::Tensor &h_post,
        const c10::optional<at::Tensor> &gamma, double hc_eps, const c10::optional<at::Tensor> &grad_x_post)
{
    TORCH_CHECK(x.numel() > 0, "Tensor x is empty.");
    TORCH_CHECK(phi.numel() > 0, "Tensor phi is empty.");
    TORCH_CHECK(alpha.numel() > 0, "Tensor alpha is empty.");
    TORCH_CHECK(grad_h_in.numel()  > 0, "Tensor grad_h_in is empty");
    TORCH_CHECK(grad_h_post.numel() > 0, "Tensor grad_h_post is empty.");
    TORCH_CHECK(grad_h_res.numel() > 0, "Tensor grad_h_res is empty.");
    TORCH_CHECK(inv_rms.numel() > 0, "Tensor inv_rms is empty.");
    TORCH_CHECK(h_mix.numel() > 0, "Tensor h_mix is empty.");
    TORCH_CHECK(h_pre.numel() > 0, "Tensor h_pre is empty.");
    TORCH_CHECK(h_post.numel() > 0, "Tensor h_post is empty.");
    TORCH_CHECK(x.dim() == TND_DIMS || x.dim() == BSND_DIMS, "Input x must be 3D or 4D, but got ", x.dim(), "D.");
    check_mhc_pre_backward_supported();

    c10::TensorOptions xGradOptions = x.options().dtype(x.dtype());
    c10::TensorOptions fp32Options = grad_h_in.options().dtype(at::kFloat);

    at::Tensor xGrad;
    at::Tensor phiGrad;
    at::Tensor alphaGrad;
    at::Tensor biasPostGrad;
    at::Tensor gammaGrad;

    // 根据grad_h_in的维度判断shape是BSND格式或TND格式
    if (grad_h_in.dim() == NUM_THREE) {
        // BSND格式: grad_h_in [B,S,D], grad_h_post [B,S,N], grad_h_res [B,S,N,N]
        auto batch = grad_h_in.size(DIM_0);
        auto sequence = grad_h_in.size(DIM_1);
        auto dimen = grad_h_in.size(DIM_2);
        auto numsResidual = grad_h_post.size(DIM_2);

        // 4维输入场景（BSND格式）
        // xGrad: [B, S, n, D]
        // phiGrad: [n^2 + 2n, nD]
        // biasPostGrad: [n^2 + 2n]
        // xGrad: [B, S, n, D]
        // gammaGrad: [n, D]
        c10::SmallVector<int64_t, BSND_DIMS> xGradSize = {batch, sequence, numsResidual, dimen};
        c10::SmallVector<int64_t, NUM_TWO> phiGradSize = {(2 * numsResidual) + (numsResidual * numsResidual), numsResidual * dimen};
        c10::SmallVector<int64_t, NUM_ONE> alphaGradSize = {3};
        c10::SmallVector<int64_t, NUM_ONE> biasPostGradSize = {(2 * numsResidual) + (numsResidual * numsResidual)};
        c10::SmallVector<int64_t, NUM_ONE> gammaGradSize = {numsResidual, dimen};

        xGrad = at::empty(xGradSize, xGradOptions);
        phiGrad = at::empty(phiGradSize, fp32Options);
        alphaGrad = at::empty(alphaGradSize, fp32Options);
        biasPostGrad = at::empty(biasPostGradSize, fp32Options);
        gammaGrad = at::empty(gammaGradSize, fp32Options);

    } else {
        // TND格式: grad_h_in [T,D], grad_h_post [T,N], grad_h_res [T,N,N]
        auto t = grad_h_in.size(DIM_0);
        auto dimen = grad_h_in.size(DIM_1);
        auto numsResidual = grad_h_post.size(DIM_1);

        // 3维输入场景（TND格式）
        // xGrad: [T, n, D]
        // phiGrad: [n^2 + 2n, nD]
        // biasPostGrad: [n^2 + 2n]
        // xGrad: [B, S, n, D]
        // gammaGrad: [n, D]
        c10::SmallVector<int64_t, TND_DIMS> xGradSize = {t, numsResidual, dimen};
        c10::SmallVector<int64_t, NUM_TWO> phiGradSize = {(2 * numsResidual) + (numsResidual * numsResidual), numsResidual * dimen};
        c10::SmallVector<int64_t, NUM_ONE> alphaGradSize = {3};
        c10::SmallVector<int64_t, NUM_ONE> biasPostGradSize = {(2 * numsResidual) + (numsResidual * numsResidual)};
        c10::SmallVector<int64_t, NUM_ONE> gammaGradSize = {numsResidual, dimen};

        xGrad = at::empty(xGradSize, xGradOptions);
        phiGrad = at::empty(phiGradSize, fp32Options);
        alphaGrad = at::empty(alphaGradSize, fp32Options);
        biasPostGrad = at::empty(biasPostGradSize, fp32Options);
        gammaGrad = at::empty(gammaGradSize, fp32Options);
    }

    EXEC_NPU_CMD(aclnnMhcPreBackward, x, phi, alpha, grad_h_in, grad_h_post,
            grad_h_res, inv_rms, h_mix, h_pre, h_post, gamma, grad_x_post, hc_eps, xGrad, phiGrad, alphaGrad, biasPostGrad, gammaGrad);

    return std::make_tuple(xGrad, phiGrad, alphaGrad, biasPostGrad, gammaGrad);
}

} // namespace op_api
