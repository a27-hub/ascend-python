#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/WrapDimUtils.h>
#include <c10/core/Scalar.h>
#include <torch/library.h>

#include <numeric>
#include <utility>

namespace op_api {

at::Tensor renorm_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const Scalar& p,
    int64_t dim,
    const Scalar& maxnorm) {
    
  auto n = self.dim();
  dim = c10::maybe_wrap_dim(dim, n);
  auto reduce_dims = at::DimVector(n);
  std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
  reduce_dims.erase(reduce_dims.begin() + dim);

  // NPU: BFloat16/Half/float → float
  auto scalar_type = self.scalar_type();
  auto acc_type = (scalar_type == c10::ScalarType::BFloat16 ||
                   scalar_type == c10::ScalarType::Half ||
                   scalar_type == c10::ScalarType::Float)
                  ? c10::ScalarType::Float
                  : scalar_type;

  auto norm = at::linalg_vector_norm(
      self, p, reduce_dims, /*keepdim=*/true, /*dtype=*/acc_type);

  const auto real_acc_type = c10::toRealValueType(acc_type);
  auto grad_output = (self.conj() * grad);
  
  if (real_acc_type != acc_type) {
    grad_output = at::real(grad_output);
  }
  
  grad_output = grad_output.sum(
      reduce_dims, /*keepdim=*/true, /*dtype=*/real_acc_type);
      
  auto nb = at::linalg_vector_norm_backward(
      std::move(grad_output), self, p, norm, reduce_dims, /*keepdim=*/true);

  auto invnorm = (norm + 1e-7).reciprocal();
  auto grad_norm = maxnorm * invnorm * (grad - invnorm * nb);
  
  return at::where(norm > maxnorm, grad_norm.to(grad.scalar_type()), grad);
}

} // namespace op_api

// ============ 注册到 NPU 后端 ============
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("renorm_backward", &op_api::renorm_backward);
}