#include <torch/library.h>

#include <ATen/Tensor.h>
#include <ATen/ATen.h>

#include "op_plugin/include/ops.h"

// RegisterNPU.cpp
namespace at {
namespace {
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
m.impl("matmul",
TORCH_FN(at::npu::wrapper_NPU__matmul_overadpt));
m.impl("matmul.out",
TORCH_FN(at::npu::wrapper_NPU_out_matmul_out_overadpt));
m.impl("matmul_backward",
TORCH_FN(at::npu::wrapper_NPU__matmul_backward_overadpt));
m.impl("matmul_double_backward",
TORCH_FN(at::npu::wrapper_NPU__matmul_double_backward_overadpt));
};
}
}