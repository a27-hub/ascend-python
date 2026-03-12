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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t DIM_3 = 3;
const static int64_t BSND_DIMS = 4;
const static int64_t TND_DIMS = 4;


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_mhc_pre(
        const at::Tensor &x, const at::Tensor &phi,
        const at::Tensor &alpha, const at::Tensor &bias,
        const c10::optional<at::Tensor> &gamma, 
        double norm_eps, double hc_eps, int64_t out_flag)
{
    TORCH_CHECK(x.numel() > 0, "Tensor x is empty.");
    TORCH_CHECK(phi.numel() > 0, "Tensor phi is empty..");
    TORCH_CHECK(alpha.numel() > 0, "Tensor alpha is empty.");
    TORCH_CHECK(bias.numel() > 0, "Tensor bias is empty.");

    c10::TensorOptions hInOptions = x.options().dtype(x.dtype());
    c10::TensorOptions hOptions = x.options().dtype(at::kFloat);
    auto matK = phi.size(DIM_0);

    at::Tensor outHin;
    at::Tensor outHpost;
    at::Tensor outHres;
    at::Tensor outInvRms;
    at::Tensor mmRes;
    at::Tensor outHPre;
    at::Tensor null_Tensor;

    if (x.dim() == 4) {
        auto batch = x.size(DIM_0);
        auto sequence = x.size(DIM_1);
        auto numsResidual = x.size(DIM_2);
        auto dimen = x.size(DIM_3);
            

        c10::SmallVector<int64_t, BSND_DIMS - 1> outHinSize;
        c10::SmallVector<int64_t, BSND_DIMS - 1> outHpostSize;
        c10::SmallVector<int64_t, BSND_DIMS> outHresSize;
        c10::SmallVector<int64_t, BSND_DIMS - 2> outInvRmsSize;
        c10::SmallVector<int64_t, BSND_DIMS - 1> mmResSize;
        c10::SmallVector<int64_t, BSND_DIMS - 1> outHPreSize;

        outHinSize.push_back(batch);
        outHinSize.push_back(sequence);
        outHinSize.push_back(dimen);

        outHpostSize.push_back(batch);
        outHpostSize.push_back(sequence);
        outHpostSize.push_back(numsResidual);

        outHresSize.push_back(batch);
        outHresSize.push_back(sequence);
        outHresSize.push_back(numsResidual);
        outHresSize.push_back(numsResidual);

        outInvRmsSize.push_back(batch);
        outInvRmsSize.push_back(sequence);

        mmResSize.push_back(batch);
        mmResSize.push_back(sequence);
        mmResSize.push_back(matK);

        outHPreSize.push_back(batch);
        outHPreSize.push_back(sequence);
        outHPreSize.push_back(numsResidual);
    
        outHin = at::empty(outHinSize, hInOptions);
        outHpost = at::empty(outHpostSize, hOptions);
        outHres = at::empty(outHresSize, hOptions);
        outInvRms = at::empty(outInvRmsSize, hOptions);
        mmRes = at::empty(mmResSize, hOptions);
        outHPre = at::empty(outHPreSize, hOptions);
    } else {
        auto t = x.size(DIM_0);
        auto numsResidual = x.size(DIM_1);
        auto dimen = x.size(DIM_2);
        c10::SmallVector<int64_t, TND_DIMS - 1> outHinSize;
        c10::SmallVector<int64_t, TND_DIMS - 1> outHpostSize;
        c10::SmallVector<int64_t, TND_DIMS> outHresSize;
        c10::SmallVector<int64_t, TND_DIMS - 2> outInvRmsSize;
        c10::SmallVector<int64_t, TND_DIMS - 1> mmResSize;
        c10::SmallVector<int64_t, TND_DIMS - 1> outHPreSize;

        outHinSize.push_back(t);
        outHinSize.push_back(dimen);

        outHpostSize.push_back(t);
        outHpostSize.push_back(numsResidual);

        outHresSize.push_back(t);
        outHresSize.push_back(numsResidual);
        outHresSize.push_back(numsResidual);

        outInvRmsSize.push_back(t);

        mmResSize.push_back(t);
        mmResSize.push_back(matK);

        outHPreSize.push_back(t);
        outHPreSize.push_back(numsResidual);

        outHin = at::empty(outHinSize, hInOptions);
        outHpost = at::empty(outHpostSize, hOptions);
        outHres = at::empty(outHresSize, hOptions);

        outInvRms = at::empty(outInvRmsSize, hOptions);
        mmRes = at::empty(mmResSize, hOptions);
        outHPre = at::empty(outHPreSize, hOptions);
    }
    if (out_flag) {
        EXEC_NPU_CMD(aclnnMhcPre, x, phi,
                alpha, bias, gamma, norm_eps, hc_eps, outHin, outHpost, outHres, outInvRms, mmRes, outHPre);
    } else {

        EXEC_NPU_CMD(aclnnMhcPre, x, phi,
                alpha, bias, gamma, norm_eps, hc_eps, outHin, outHpost, outHres, null_Tensor, null_Tensor, null_Tensor);       
    }

    return std::make_tuple(outHin, outHpost, outHres, outInvRms, mmRes, outHPre); // mm_res 为h_mix
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_mhc_pre_backward(
        const at::Tensor &x, const at::Tensor &phi, const at::Tensor &alpha, 
        const at::Tensor &grad_h_in, const at::Tensor &grad_h_post, const at::Tensor &grad_h_res,
        const at::Tensor &inv_rms, const at::Tensor &h_mix, const at::Tensor &h_pre, const at::Tensor &h_post,
        const c10::optional<at::Tensor> &gamma, double hc_eps)
{
    TORCH_CHECK(x.numel() > 0, "Tensor x is empty.");
    TORCH_CHECK(phi.numel() > 0, "Tensor phi is empty.");
    TORCH_CHECK(alpha.numel() > 0, "Tensor alpha is empty.");
    TORCH_CHECK(grad_h_in.numel()  > 0, "Tensor grad_h_in dims is empty");
    TORCH_CHECK(grad_h_post.numel() > 0, "Tensor grad_h_post is empty.");
    TORCH_CHECK(grad_h_res.numel() > 0, "Tensor grad_h_res is empty.");
    TORCH_CHECK(inv_rms.numel() > 0, "Tensor inv_rms is empty.");
    TORCH_CHECK(h_mix.numel() > 0, "Tensor h_mix is empty.");
    TORCH_CHECK(h_pre.numel() > 0, "Tensor h_pre is empty.");
    TORCH_CHECK(h_post.numel() > 0, "Tensor h_post is empty.");

    c10::TensorOptions xGradOptions = grad_h_in.options().dtype(grad_h_in.dtype());
    c10::TensorOptions fp32Options = grad_h_in.options().dtype(at::kFloat);

    at::Tensor xGrad;
    at::Tensor hcWeightGrad;
    at::Tensor alphaGrad;
    at::Tensor biasPostGrad;
    at::Tensor gammaGrad;

    if (grad_h_in.dim() == 3) {
        // BSND格式: grad_h_in [B,S,D], grad_h_post [B,S,N], grad_h_res [B,S,N,N]
        auto batch = grad_h_in.size(DIM_0);
        auto sequence = grad_h_in.size(DIM_1);
        auto dimen = grad_h_in.size(DIM_2);
        auto numsResidual = grad_h_post.size(DIM_2);

        c10::SmallVector<int64_t, BSND_DIMS> xGradSize = {batch, sequence, numsResidual, dimen};
        c10::SmallVector<int64_t, 2> hcWeightGradSize = {(2 * numsResidual) + (numsResidual * numsResidual), numsResidual * dimen};
        c10::SmallVector<int64_t, 1> alphaGradSize = {3};
        c10::SmallVector<int64_t, 1> biasPostGradSize = {(2 * numsResidual) + (numsResidual * numsResidual)};
        c10::SmallVector<int64_t, 1> gammaGradSize = {numsResidual, dimen};

        xGrad = at::empty(xGradSize, xGradOptions);
        hcWeightGrad = at::empty(hcWeightGradSize, fp32Options);
        alphaGrad = at::empty(alphaGradSize, fp32Options);
        biasPostGrad = at::empty(biasPostGradSize, fp32Options);
        gammaGrad = at::empty(gammaGradSize, fp32Options);

    } else {
        // TND格式: grad_h_in [T,D], grad_h_post [T,N], grad_h_res [T,N,N]
        auto t = grad_h_in.size(DIM_0);
        auto dimen = grad_h_in.size(DIM_1);
        auto numsResidual = grad_h_post.size(DIM_1);

        c10::SmallVector<int64_t, TND_DIMS> xGradSize = {t, numsResidual, dimen};
        c10::SmallVector<int64_t, 2> hcWeightGradSize = {(2 * numsResidual) + (numsResidual * numsResidual), numsResidual * dimen};
        c10::SmallVector<int64_t, 1> alphaGradSize = {3};
        c10::SmallVector<int64_t, 1> biasPostGradSize = {(2 * numsResidual) + (numsResidual * numsResidual)};
        c10::SmallVector<int64_t, 1> gammaGradSize = {numsResidual, dimen};

        xGrad = at::empty(xGradSize, xGradOptions);
        hcWeightGrad = at::empty(hcWeightGradSize, fp32Options);
        alphaGrad = at::empty(alphaGradSize, fp32Options);
        biasPostGrad = at::empty(biasPostGradSize, fp32Options);
        gammaGrad = at::empty(gammaGradSize, fp32Options);
    }

    //TODO 修改hcWeightGrad对应为aclnn接口的gradPhi， 本文件的hcWeightGrad改为phiGrad
    EXEC_NPU_CMD(aclnnMhcPreBackward, x, phi, alpha, grad_h_in, grad_h_post,
            grad_h_res, inv_rms, h_mix, h_pre, h_post, gamma, hc_eps, xGrad, hcWeightGrad, alphaGrad, biasPostGrad, gammaGrad);

    return std::make_tuple(xGrad, hcWeightGrad, alphaGrad, biasPostGrad, gammaGrad);
}

}
