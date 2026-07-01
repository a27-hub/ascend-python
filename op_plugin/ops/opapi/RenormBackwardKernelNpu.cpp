#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/WrapDimUtils.h>
#include <c10/core/Scalar.h>
#include <torch/library.h>

#include <numeric>
#include <utility>

namespace op_api {

// ============================================================
// 辅助函数：unsqueeze_multiple
// ============================================================
static Tensor unsqueeze_multiple(
    const Tensor& t,
    OptionalIntArrayRef opt_dim,
    size_t n_dims) {
  if (opt_dim.has_value()) {
    IntArrayRef dim = opt_dim.value();
    auto dim_size = dim.size();
    if (dim_size == 0) {
      return t;
    } else if (dim_size == 1) {
      return t.unsqueeze(dim[0]);
    }
  }
  auto dims_to_unsqueeze = at::dim_list_to_bitset(opt_dim, n_dims);
  Tensor res = t;
  for (const auto i : c10::irange(n_dims)) {
    if (dims_to_unsqueeze[i]) {
      res = res.unsqueeze(static_cast<int64_t>(i));
    }
  }
  return res;
}

// ============================================================
// norm_backward 实现
// ============================================================
Tensor norm_backward(
    const Tensor& grad,
    const Tensor& self,
    const std::optional<Scalar>& p_,
    const Tensor& norm) {
  return norm_backward(grad, self, p_, norm, {}, true);
}

Tensor norm_backward(
    Tensor grad,
    const Tensor& self,
    const std::optional<Scalar>& p_,
    Tensor norm,
    IntArrayRef dim,
    bool keepdim) {
  size_t ndim = self.dim();
  double p = p_.value_or(2.0).toDouble();
  Tensor self_scaled;
  Tensor scale_v;

  if (!keepdim && self.dim() != 0) {
    grad = unsqueeze_multiple(grad, dim, ndim);
    norm = unsqueeze_multiple(norm, dim, ndim);
  }

  if (p == 0.0) {
    return {};
  } else if (p == 1.0) {
    return self.sgn() * grad;
  } else if (p == 2.0) {
    return grad * (self / norm).masked_fill_(norm == 0, 0);
  } else if (std::isinf(p)) {
    auto self_abs = self.abs();
    auto mask = self_abs.eq(norm).logical_or(self_abs.isnan());
    return self.sgn() * ((grad / mask.sum(dim, true)) * mask);
  } else if (p < 1.0) {
    self_scaled =
        self.sgn() * self.abs().pow_(p - 1).masked_fill_(self == 0, 0);
    return self_scaled * grad * norm.pow(1 - p);
  } else if (p < 2.0) {
    self_scaled = self.sgn() * self.abs().pow_(p - 1);
    scale_v = grad / norm.pow(p - 1);
    scale_v.masked_fill_(norm == 0, 0);
    return self_scaled * scale_v;
  } else {
    self_scaled = self * self.abs().pow_(p - 2);
    scale_v = grad / norm.pow(p - 1);
    scale_v.masked_fill_(norm == 0, 0);
    return self_scaled * scale_v;
  }
}

// ============================================================
// npu_renorm_backward 主逻辑
// ============================================================
at::Tensor npu_renorm_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const c10::Scalar& p,
    int64_t dim,
    const c10::Scalar& maxnorm) {
    
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

  // 调用同文件中的 norm_backward
  auto nb = norm_backward(
      std::move(grad_output), self, p, norm, reduce_dims, /*keepdim=*/true);

  auto invnorm = (norm + 1e-7).reciprocal();
  auto grad_norm = maxnorm * invnorm * (grad - invnorm * nb);
  
  return at::where(norm > maxnorm, grad_norm.to(grad.scalar_type()), grad);
}

} // namespace op_api