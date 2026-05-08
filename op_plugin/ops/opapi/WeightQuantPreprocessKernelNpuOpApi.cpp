// Copyright (c) 2026 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/core/NPUBridge.h"
#include <unordered_map>

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
constexpr int64_t NZ_16 = 16;
constexpr int64_t NZ_C0_32_FOR_B4 = 16;

enum class QuantDataFlow { UNKNOWN = 0, MM_MX_A8W4 = 1, GMM_MX_A8W4 = 2 };

struct QuantContext {
    const at::Tensor& weight;
    const at::Tensor& weight_scale;
    const c10::optional<at::Tensor>& weight_offset;
    const c10::optional<at::Tensor>& bias;
    int64_t x_dtype;
    int64_t weight_dtype;
    c10::optional<int64_t> x_scale_dtype;
    int64_t weight_scale_dtype;
    int64_t k_group_size;
    c10_npu::SocVersion soc_version;
    QuantDataFlow data_flow;
    bool is_gmm = false;
    
    at::Tensor out_weight;
    at::Tensor out_weight_scale;
    at::Tensor out_weight_offset;
    at::Tensor out_bias;
};

using PrepareFunc = void (*)(QuantContext& ctx);
using DataFlowJudgeFunc = void (*)(QuantContext& ctx);

inline int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

void set_tensor_format_and_storage_shape(at::Tensor& tensor, int64_t format, 
    const c10::SmallVector<int64_t, 5>& storage_shape) {
    auto& desc = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_;
    desc.npu_format_ = static_cast<aclFormat>(format);
    desc.storage_sizes_ = storage_shape;
}

void judge_mm_mx_a8w4(QuantContext& ctx) {
    aclDataType x_acl_dtype = c10_npu::GetAclDataType(ctx.x_dtype);
    aclDataType weight_acl_dtype = c10_npu::GetAclDataType(ctx.weight_dtype);
    aclDataType x_scale_acl_dtype = ctx.x_scale_dtype.has_value() ? 
        c10_npu::GetAclDataType(ctx.x_scale_dtype.value()) : ACL_DT_UNDEFINED;
    aclDataType weight_scale_acl_dtype = c10_npu::GetAclDataType(ctx.weight_scale_dtype);
    
    bool dtype_match = (x_acl_dtype == ACL_FLOAT8_E4M3FN) &&
        (weight_acl_dtype == ACL_FLOAT4_E2M1) &&
        (x_scale_acl_dtype == ACL_FLOAT8_E8M0) &&
        (weight_scale_acl_dtype == ACL_FLOAT8_E8M0);
    if (dtype_match && ctx.weight.dim() == 2) {
        ctx.data_flow = QuantDataFlow::MM_MX_A8W4;
    }
}

void judge_gmm_mx_a8w4(QuantContext& ctx) {
    aclDataType x_acl_dtype = c10_npu::GetAclDataType(ctx.x_dtype);
    aclDataType weight_acl_dtype = c10_npu::GetAclDataType(ctx.weight_dtype);
    aclDataType x_scale_acl_dtype = ctx.x_scale_dtype.has_value() ? 
        c10_npu::GetAclDataType(ctx.x_scale_dtype.value()) : ACL_DT_UNDEFINED;
    aclDataType weight_scale_acl_dtype = c10_npu::GetAclDataType(ctx.weight_scale_dtype);
    
    bool dtype_match = (x_acl_dtype == ACL_FLOAT8_E4M3FN) &&
        (weight_acl_dtype == ACL_FLOAT4_E2M1) &&
        (x_scale_acl_dtype == ACL_FLOAT8_E8M0) &&
        (weight_scale_acl_dtype == ACL_FLOAT8_E8M0);
    if (dtype_match && ctx.weight.dim() == 3) {
        ctx.is_gmm = true;
        ctx.data_flow = QuantDataFlow::GMM_MX_A8W4;
    }
}

static void prepare_out_weight_nz_c0_32_nk(QuantContext& ctx)
{
    auto weight_view_shape = op_infer::array_to_small_vector(ctx.weight.sizes());
    ctx.out_weight = npu_preparation::apply_tensor_without_format(weight_view_shape, ctx.weight.options());
    
    auto weight_sizes = ctx.weight.sizes();
    int64_t weight_dim = ctx.weight.dim();
    int64_t K = weight_sizes[weight_dim - 2];
    int64_t N = weight_sizes[weight_dim - 1];
    
    c10::SmallVector<int64_t, 5> storage_shape;
    if (ctx.is_gmm) {
        int64_t G = weight_sizes[0];
        storage_shape = {G, ceil_div(K, NZ_C0_32_FOR_B4), ceil_div(N, NZ_16), NZ_16, NZ_C0_32_FOR_B4};
    } else {
        storage_shape = {ceil_div(K, NZ_C0_32_FOR_B4), ceil_div(N, NZ_16), NZ_16, NZ_C0_32_FOR_B4};
    }
    
    set_tensor_format_and_storage_shape(ctx.out_weight, ACL_FORMAT_FRACTAL_NZ_C0_16, storage_shape);
}

static void prepare_out_weight_scale(QuantContext& ctx)
{
    auto scale_sizes = op_infer::array_to_small_vector(ctx.weight_scale.sizes());
    ctx.out_weight_scale = npu_preparation::apply_tensor_without_format(scale_sizes, ctx.weight_scale.options());
}

static void prepare_out_weight_offset(QuantContext& ctx)
{
    if (ctx.weight_offset.has_value() && ctx.weight_offset.value().defined()) {
        auto offset_sizes = op_infer::array_to_small_vector(ctx.weight_offset.value().sizes());
        ctx.out_weight_offset = npu_preparation::apply_tensor_without_format(
            offset_sizes, ctx.weight_offset.value().options());
    }
}

static void prepare_out_bias(QuantContext& ctx)
{
    if (ctx.bias.has_value() && ctx.bias.value().defined()) {
        auto bias_sizes = op_infer::array_to_small_vector(ctx.bias.value().sizes());
        ctx.out_bias = npu_preparation::apply_tensor_without_format(bias_sizes, ctx.bias.value().options());
    }
}

static void execute_prepares(const std::vector<PrepareFunc>& prepares, QuantContext& ctx)
{
    for (const auto& prepare : prepares) {
        prepare(ctx);
    }
}

static const std::unordered_map<c10_npu::SocVersion, std::vector<DataFlowJudgeFunc>> SOC_VERSION_DATA_FLOW_MAP = {
    {c10_npu::SocVersion::Ascend950, {judge_mm_mx_a8w4, judge_gmm_mx_a8w4}},
};

static const std::unordered_map<c10_npu::SocVersion, std::unordered_map<QuantDataFlow, std::vector<PrepareFunc>>> SOC_DATA_FLOW_PREPARE_MAP = {
    {c10_npu::SocVersion::Ascend950, {
        {QuantDataFlow::MM_MX_A8W4, {
            prepare_out_weight_nz_c0_32_nk,
            prepare_out_weight_scale,
            prepare_out_weight_offset,
            prepare_out_bias
        }},
        {QuantDataFlow::GMM_MX_A8W4, {
            prepare_out_weight_nz_c0_32_nk,
            prepare_out_weight_scale,
            prepare_out_weight_offset,
            prepare_out_bias
        }}
    }},
};

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_weight_quant_preprocess(
    const at::Tensor& weight,
    const at::Tensor& weight_scale,
    int64_t x_dtype,
    int64_t weight_dtype,
    int64_t weight_scale_dtype,
    const c10::optional<at::Tensor>& weight_offset,
    const c10::optional<at::Tensor>& bias,
    c10::optional<int64_t> x_scale_dtype,
    c10::optional<int64_t> k_group_size)
{
    auto soc_version = c10_npu::GetSocVersion();
    auto it = SOC_VERSION_DATA_FLOW_MAP.find(soc_version);
    TORCH_CHECK(it != SOC_VERSION_DATA_FLOW_MAP.end(),
        "Unsupported NPU architecture: " + std::to_string(static_cast<int>(soc_version)) + OPS_ERROR(ErrCode::NOT_SUPPORT));
    
    int64_t k_group_size_real = k_group_size.value_or(0);
    
    QuantContext ctx{
        weight,
        weight_scale,
        weight_offset,
        bias,
        x_dtype,
        weight_dtype,
        x_scale_dtype,
        weight_scale_dtype,
        k_group_size_real,
        soc_version,
        QuantDataFlow::UNKNOWN,
        false,
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor()
    };
    
    for (const auto& judge : it->second) {
        judge(ctx);
        if (ctx.data_flow != QuantDataFlow::UNKNOWN) break;
    }
    TORCH_CHECK(ctx.data_flow != QuantDataFlow::UNKNOWN,
        "Unsupported data flow combination." + OPS_ERROR(ErrCode::PARAM));
    
    auto arch_it = SOC_DATA_FLOW_PREPARE_MAP.find(soc_version);
    TORCH_CHECK(arch_it != SOC_DATA_FLOW_PREPARE_MAP.end(),
        "No prepare functions registered for SocVersion: " + std::to_string(static_cast<int>(soc_version)) + OPS_ERROR(ErrCode::PARAM));
    
    auto data_flow_it = arch_it->second.find(ctx.data_flow);
    TORCH_CHECK(data_flow_it != arch_it->second.end(),
        "No prepare functions registered for DataFlow: " + std::to_string(static_cast<int>(ctx.data_flow)) + OPS_ERROR(ErrCode::PARAM));
    
    execute_prepares(data_flow_it->second, ctx);
    
    TensorWrapper weight_wrapper = make_wrapper(ctx.out_weight, 
        c10::optional<int64_t>(ctx.weight_dtype));
    TensorWrapper scale_wrapper = make_wrapper(ctx.out_weight_scale,
        c10::optional<int64_t>(ctx.weight_scale_dtype));
    
    aclDataType x_acl_type = c10_npu::GetAclDataType(x_dtype);
    aclDataType x_scale_acl_type =
        x_scale_dtype.has_value() ? c10_npu::GetAclDataType(x_scale_dtype.value()) : ACL_DT_UNDEFINED;

    at::Tensor weight_offset_tensor = weight_offset.value_or(at::Tensor());
    at::Tensor bias_tensor = bias.value_or(at::Tensor());
    EXEC_NPU_CMD(aclnnWeightQuantPreprocess,
        weight, weight_scale,
        weight_offset_tensor,
        bias_tensor,
        x_acl_type, x_scale_acl_type, ctx.k_group_size,
        weight_wrapper, scale_wrapper,
        ctx.out_weight_offset,
        ctx.out_bias);
    
    return std::make_tuple(ctx.out_weight, ctx.out_weight_scale, ctx.out_weight_offset, ctx.out_bias);
}

} // namespace op_api