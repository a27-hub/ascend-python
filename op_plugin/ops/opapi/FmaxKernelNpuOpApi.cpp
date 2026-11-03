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
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

using npu_preparation = at_npu::native::OpPreparation;

namespace op_api {

// fmax(self, other) = where(isnan(other) | (other < self), self, other)
// 组合语义对齐 PyTorch 官方 decomposition（torch/_inductor/decomposition.py::fmax）。
// CANN 无 aclnnIsNan，isnan 以 NeTensor(x, x)（x != x）表达；浮点路径共 4 次 launch：
// NeTensor + LtTensor + LogicalOr + SWhere（混合 dtype 由 aclnn 插 Cast，6~8 次 launch）。
// 整数/bool 无 NaN，直接 aclnnMaximum（与 CUDA 整数路径同构，见 MaxMinElementwiseKernel.cu）。

inline bool _fmax_combo_symbols_available()
{
    static const bool available = check_aclnn_kernel_available("aclnnNeTensor") &&
        check_aclnn_kernel_available("aclnnLtTensor") &&
        check_aclnn_kernel_available("aclnnLogicalOr") &&
        check_aclnn_kernel_available("aclnnSWhere") &&
        check_aclnn_kernel_available("aclnnMaximum");
    return available;
}

// complex 必须在任何设备搬运之前报错，文案与上游结构化 kernel 一致
// （aten/src/ATen/native/BinaryOps.cpp TORCH_META_FUNC(fmax)）。
inline void _fmax_check_complex(const at::Tensor &self, const at::Tensor &other)
{
    TORCH_CHECK_TYPE(
        !self.is_complex() && !other.is_complex(),
        "fmax not implemented for complex tensors.");
}

inline bool _fmax_fallback_condition(const at::Tensor &self, const at::Tensor &other)
{
    // 不探测 aclnnFmax：CANN 原生算子发布后按编译期能力分支收敛（同一份代码按 CANN
    // 版本分别构建），不做运行时 dlsym 探测。
    static const bool is_support_soc =
        (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
            c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
        (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_soc || !_fmax_combo_symbols_available()) {
        TORCH_NPU_WARN_ONCE(
            "CAUTION: The operator aten::fmax and aten::fmax.out is currently not supported on "
            "the NPU backend. Now this operator will fallback to run on the CPU and may have "
            "performance implications.");
        return true;
    }
    // dtype 门控：提升后为 double 时回 CPU（实测组合接口支持 double，但 ≈1.3M 元素以上
    // 慢于 fallback，本期不做原生）。complex 已在前置校验中报错，不会走到这里。
    const at::ScalarType high_type = at::native::result_type(self, other);
    if (high_type == at::ScalarType::Double) {
        TORCH_NPU_WARN_ONCE(
            "CAUTION: The operator aten::fmax and aten::fmax.out with input dtype ",
            at::toString(high_type),
            " is currently not supported on the NPU backend. Now this operator will fallback "
            "to run on the CPU and may have performance implications.");
        return true;
    }
    return false;
}

// 浮点组合实现（4 launch）。mask 形状（原输入形状）总可广播到 output_size；
// 中间 mask 为独立 tensor，不别名输入；result 可与 self/other 复用同一块内存
// （SWhere 的 out 允许与输入共享），已由 test_fmax_out_same_tensor 覆盖。
at::Tensor &fmax_combo_out(
    const at::Tensor &self,
    const at::Tensor &other,
    at::Tensor &result,
    const at::IntArrayRef &output_size)
{
    at::Tensor nan_mask = npu_preparation::apply_tensor_without_format(
        other.sizes(), other.options().dtype(at::kBool));
    EXEC_NPU_CMD(aclnnNeTensor, other, other, nan_mask);  // isnan(other)
    at::Tensor lt_mask = npu_preparation::apply_tensor_without_format(
        output_size, self.options().dtype(at::kBool));
    EXEC_NPU_CMD(aclnnLtTensor, other, self, lt_mask);    // other < self（广播输出）
    at::Tensor mask = npu_preparation::apply_tensor_without_format(
        output_size, self.options().dtype(at::kBool));
    EXEC_NPU_CMD(aclnnLogicalOr, nan_mask, lt_mask, mask);
    EXEC_NPU_CMD(aclnnSWhere, mask, self, other, result);
    return result;
}

// 按提升后的 dtype 选路：浮点走组合，整数/bool 走 aclnnMaximum（1 launch）。
inline at::Tensor &_fmax_launch(
    const at::Tensor &self,
    const at::Tensor &other,
    at::Tensor &result,
    at::ScalarType result_dtype,
    const at::IntArrayRef &output_size)
{
    if (c10::isFloatingType(result_dtype)) {
        return fmax_combo_out(self, other, result, output_size);
    }
    EXEC_NPU_CMD(aclnnMaximum, self, other, result);  // 整数/bool
    return result;
}

at::Tensor &fmax_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &result)
{
    _fmax_check_complex(self, other);
    auto [self_device, other_device] = prepare_binary_tensors(self, other);
    at::ScalarType result_dtype = at::native::result_type(self_device, other_device);
    auto output_size = op_infer::broadcast_ops_npu_output_size(self_device, other_device);
    // 校验设备 / 尺寸 / 内存重叠：尺寸不匹配按 resize + 弃用告警、部分重叠抛 RuntimeError，
    // 均与上游 out 变体一致；out dtype 不匹配按 TensorIterator 语义——可安全 cast
    // （int→int、int→float 等）按 cast 写入，不可安全 cast（→bool、float→int 等）报错。
    // 必须在 fallback 分流之前执行：op_plugin 以 PrivateUse1 直接注册实现，绕过了上游
    // TORCH_META_FUNC(fmax) 的 resize，本 kernel 需自行复刻；否则 fallback 分支里
    // result.copy_() 会因形状不可广播报错，零元素 out 时更会静默空转返回空张量。
    npu_preparation::check_tensor({self_device, other_device}, result, output_size);
    if (_fmax_fallback_condition(self, other)) {
        // 用原始输入做 CPU 计算：self/other 可能本就是 CPU 标量，.cpu() 为空操作，
        // 不应使用已搬到 device 的副本，避免多余的 H2D/D2H。
        at::Tensor self_cpu = self.cpu();
        at::Tensor other_cpu = other.cpu();
        at::Tensor result_cpu = result.cpu();
        at::fmax_outf(self_cpu, other_cpu, result_cpu);
        result.copy_(result_cpu);
        return result;
    }
    if (result.scalar_type() != result_dtype) {
        TORCH_CHECK(
            canCast(result_dtype, result.scalar_type()),
            "result type ",
            result_dtype,
            " can't be cast to the desired output type ",
            result.scalar_type(),
            OPS_ERROR(ErrCode::TYPE));
        at::Tensor tmp = npu_preparation::apply_tensor_without_format(
            output_size, self_device.options().dtype(result_dtype));
        _fmax_launch(self_device, other_device, tmp, result_dtype, output_size);
        result.copy_(tmp);
        return result;
    }
    return _fmax_launch(self_device, other_device, result, result_dtype, output_size);
}

at::Tensor fmax(const at::Tensor &self, const at::Tensor &other)
{
    _fmax_check_complex(self, other);
    if (_fmax_fallback_condition(self, other)) {
        at::Tensor self_cpu = self.cpu();
        at::Tensor other_cpu = other.cpu();
        at::Device result_device =
            torch_npu::utils::is_npu(self) ? self.device() : other.device();
        return at::fmax(self_cpu, other_cpu).to(result_device);
    }
    auto [self_device, other_device] = prepare_binary_tensors(self, other);
    at::ScalarType result_dtype = at::native::result_type(self_device, other_device);
    auto output_size = op_infer::broadcast_ops_npu_output_size(self_device, other_device);
    at::Tensor result = npu_preparation::apply_tensor_without_format(
        output_size, self_device.options().dtype(result_dtype));
    _fmax_launch(self_device, other_device, result, result_dtype, output_size);
    return result;
}

}  // namespace op_api
