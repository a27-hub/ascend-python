#include <ATen/ATen.h>
#include <torch/extension.h>
#include "op_plugin/include/atb_ops.h"

namespace {

void check_npu_tensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        name,
        " must be an NPU tensor, got device ",
        tensor.device());
}

at::Tensor quantize_per_tensor(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& zero_point)
{
    check_npu_tensor(x, "x");
    check_npu_tensor(scale, "scale");
    check_npu_tensor(zero_point, "zero_point");

    at::Tensor y = at::zeros_like(x, x.options().dtype(at::kChar));
    at_npu::native::atb::_npu_quantize_per_tensor(x, scale, zero_point, y);
    return y;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quantize_per_tensor", &quantize_per_tensor, "ATB quantize per tensor wrapper");
}
