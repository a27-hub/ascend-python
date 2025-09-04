// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.

#include <ATen/native/LinearAlgebraUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
std::tuple<at::Tensor &, at::Tensor &> triangular_solve_out(const at::Tensor &self, const at::Tensor &A, bool upper,
                                                            bool transpose, bool unitriangular, at::Tensor &X,
                                                            at::Tensor &M)
{
    at::Tensor self_broadcasted;
    at::Tensor a_broadcasted;
    std::tie(self_broadcasted, a_broadcasted) = at::native::_linalg_broadcast_batch_dims(self, A, "triangular_solve");
    X.resize_as_(self_broadcasted);
    M.resize_as_(a_broadcasted);
    EXEC_NPU_CMD(aclnnTriangularSolve, self, A, upper, transpose, unitriangular, X, M);
    return std::tie(X, M);
}

} // namespace acl_op
