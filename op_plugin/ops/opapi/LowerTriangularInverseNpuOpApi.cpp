#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& npu_lower_triangular_inverse(const at::Tensor& x) {

  at::Tensor result = npu_preparation::apply_tensor_without_format(x);

  EXEC_NPU_CMD(aclnnLowerTriangularInverse, x, result);
  return result;
}
}