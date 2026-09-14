#include "op_plugin/OpApiInterface.h"
#include <ATen/native/nested/NestedTensorMath.h>
#include <vector>

namespace op_api {
at::Tensor to_padded_tensor(
    const at::Tensor& self,
    double padding,
    at::OptionalSymIntArrayRef output_size) {
  // 将 SymInt[]? 转换为 OptionalIntArrayRef
  at::OptionalIntArrayRef output_size_int = c10::nullopt;
  std::vector<int64_t> output_size_vec;
  if (output_size.has_value()) {
    for (const auto& sym : output_size.value()) {
      output_size_vec.push_back(sym.expect_int());
    }
    output_size_int = at::IntArrayRef(output_size_vec);
  }
  // 调用 PyTorch 原生 CPU 实现
  return at::native::NestedTensor_to_padded_tensor_generic(
      self, padding, output_size_int);
}
} // namespace op_api