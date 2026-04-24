#include <ATen/ATen.h>
#include <torch/extension.h>

#include "op_plugin/include/atb_ops.h"

namespace {

void check_quantize_inputs(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& zero_point)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
        "input_x must be an NPU tensor, got device ", x.device());
    TORCH_CHECK(scale.device() == x.device(),
        "scale must be on the same device as input_x, got ", scale.device(), " vs ", x.device());
    TORCH_CHECK(zero_point.device() == x.device(),
        "zero_point must be on the same device as input_x, got ", zero_point.device(), " vs ", x.device());

    TORCH_CHECK(x.scalar_type() == at::kHalf,
        "input_x must be float16, got ", x.scalar_type());
    TORCH_CHECK(scale.scalar_type() == at::kHalf,
        "scale must be float16, got ", scale.scalar_type());
    TORCH_CHECK(zero_point.scalar_type() == at::kChar,
        "zero_point must be int8, got ", zero_point.scalar_type());

    TORCH_CHECK(x.dim() == 3, "input_x must be a 3D tensor, got dim=", x.dim());
    TORCH_CHECK(scale.dim() == 2, "scale must be a 2D tensor, got dim=", scale.dim());
    TORCH_CHECK(zero_point.dim() == 2, "zero_point must be a 2D tensor, got dim=", zero_point.dim());
    TORCH_CHECK(x.size(1) == scale.size(0) && x.size(2) == scale.size(1),
        "scale shape must match the trailing dimensions of input_x, got input_x=",
        x.sizes(), ", scale=", scale.sizes());
    TORCH_CHECK(scale.sizes() == zero_point.sizes(),
        "scale and zero_point must have the same shape, got ",
        scale.sizes(), " vs ", zero_point.sizes());
}

}  // namespace

at::Tensor run_atb_quantize_per_tensor(
    const at::Tensor& x,
    const at::Tensor& scale,
    const at::Tensor& zero_point)
{
    check_quantize_inputs(x, scale, zero_point);

    at::Tensor y = at::zeros(x.sizes(), x.options().dtype(at::kChar));
    at_npu::native::atb::_npu_quantize_per_tensor(x, scale, zero_point, y);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "run_atb_quantize_per_tensor",
        &run_atb_quantize_per_tensor,
        "Run at_npu::native::atb::_npu_quantize_per_tensor");
}
