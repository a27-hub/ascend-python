#include <ATen/EmptyTensor.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Operators.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <c10/util/Logging.h>
#include <torch/library.h>

#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/npu_log.h"

namespace at_npu {
namespace functionalization {
at::Tensor &add_quant_gmm(c10::DispatchKeySet dispatchKeySet, at::Tensor &self, const at::Tensor &x1,
                          const at::Tensor &x2, const at::Tensor &x2_scale, const at::Tensor &group_list,
                          const c10::optional<at::Tensor> &x1_scale, c10::optional<int64_t> group_list_type,
                          c10::OptionalArrayRef<c10::SymInt> group_sizes, c10::optional<int64_t> x1_dtype,
                          c10::optional<int64_t> x2_dtype, c10::optional<int64_t> x1_scale_dtype,
                          c10::optional<int64_t> x2_scale_dtype)
{
    // 1.对所有入参中的tensor调用sync()
    at::functionalization::impl::sync(self);
    at::functionalization::impl::sync(x1);
    at::functionalization::impl::sync(x2);
    at::functionalization::impl::sync(x2_scale);
    at::functionalization::impl::sync(group_list);
    at::functionalization::impl::sync(x1_scale);
    // 2.调用_from_functional_tensor解包所有需要调用非原地算子的入参
    auto self_unwarp = at::functionalization::impl::from_functional_tensor(self);
    auto x1_unwarp = at::functionalization::impl::from_functional_tensor(x1);
    auto x2_unwarp = at::functionalization::impl::from_functional_tensor(x2);
    auto x2_scale_unwarp = at::functionalization::impl::from_functional_tensor(x2_scale);
    auto group_list_unwarp = at::functionalization::impl::from_functional_tensor(group_list);
    auto x1_scale_unwarp = at::functionalization::impl::from_functional_tensor(x1_scale);

    // Redispath to the out-of-place op when mutable op is called by user
    at::Tensor tmp_self;
    {
        at::AutoDispatchSkipFunctionalize guard;
        // 3.调用非原地算子，得到临时输出
        tmp_self = at_npu::native::custom_ops::npu_add_quant_gmm(
            self_unwarp, x1_unwarp, x2_unwarp, x2_scale_unwarp, group_list_unwarp, x1_scale_unwarp, group_list_type,
            group_sizes, x1_dtype, x2_dtype, x1_scale_dtype, x2_scale_dtype);
    }
    // 4.将临时输出拷贝给输入，达到原地修改的目的
    at::functionalization::impl::replace_(self, tmp_self);
    at::functionalization::impl::commit_update(self);
    // 5.再次调用sync()
    at::functionalization::impl::sync(self);
    at::functionalization::impl::sync(x1);
    at::functionalization::impl::sync(x2);
    at::functionalization::impl::sync(x2_scale);
    at::functionalization::impl::sync(group_list);
    at::functionalization::impl::sync(x1_scale);

    return self;
}
}  // namespace functionalization

namespace {
// 注册Functionalize
TORCH_LIBRARY_IMPL(npu, Functionalize, m) { m.impl("npu_add_quant_gmm_", TORCH_FN(functionalization::add_quant_gmm)); }
}  // namespace

}  // namespace at_npu