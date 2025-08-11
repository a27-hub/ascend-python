#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/EmptyTensor.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/npu_log.h"
#include <c10/util/Logging.h>

#include <ATen/Operators.h>
#include <ATen/NativeFunctions.h>
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace at_npu {
namespace functionalization {
    constexpr auto exclude_keys_for_meta_dispatch =
    c10::functorch_transforms_ks |
    c10::DispatchKeySet({
        c10::DispatchKey::FuncTorchDynamicLayerBackMode,
        c10::DispatchKey::FuncTorchDynamicLayerFrontMode,
        c10::DispatchKey::Python
    });

::std::tuple<at::Tensor, at::Tensor> npu_scatter_pa_kv_cache(c10::DispatchKeySet dispatchKeySet,
                                     const at::Tensor& key,
                                     const at::Tensor& value,
                                     const at::Tensor& slot_mapping,
                                     const c10::optional<at::Tensor>& compress_lens,
                                     const c10::optional<at::Tensor>& compress_seq_offsets,
                                     const c10::optional<at::Tensor>& seq_lens,
                                     at::TensorList out)
{
    at::Tensor key_cache = out[0];
    at::Tensor value_cache = out[1];
    at::functionalization::impl::sync(key);
    at::functionalization::impl::sync(value);
    at::functionalization::impl::sync(slot_mapping);
    at::functionalization::impl::sync(compress_lens);
    at::functionalization::impl::sync(compress_seq_offsets);
    at::functionalization::impl::sync(seq_lens);
    at::functionalization::impl::sync(key_cache);
    at::functionalization::impl::sync(value_cache);

    auto key_unwarp = at::functionalization::impl::from_functional_tensor(key);
    auto value_unwarp = at::functionalization::impl::from_functional_tensor(value);
    auto slot_mapping_unwarp = at::functionalization::impl::from_functional_tensor(slot_mapping);
    auto compress_lens_unwarp = at::functionalization::impl::from_functional_tensor(compress_lens);
    auto compress_seq_offsets_unwarp = at::functionalization::impl::from_functional_tensor(compress_seq_offsets);
    auto seq_lens_unwarp = at::functionalization::impl::from_functional_tensor(seq_lens);
    auto key_cache_unwarp = at::functionalization::impl::from_functional_tensor(key_cache);
    auto value_cache_unwarp = at::functionalization::impl::from_functional_tensor(value_cache);

    at::Tensor tmp_input1;
    at::Tensor tmp_input2;
    {
        at::AutoDispatchSkipFunctionalize guard;
        auto tmp_result = at_npu::native::custom_ops::npu_scatter_pa_kv_cache_functional(key_unwarp,
                                                                                        value_unwarp,
                                                                                        key_cache_unwarp,
                                                                                        value_cache_unwarp,
                                                                                        slot_mapping_unwarp,
                                                                                        compress_lens_unwarp,
                                                                                        compress_seq_offsets_unwarp,
                                                                                        seq_lens_unwarp);
        tmp_input1 = std::get<0>(tmp_result);
        tmp_input2 = std::get<1>(tmp_result);
    }

    at::functionalization::impl::replace_(key_cache, tmp_input1);
    at::functionalization::impl::replace_(value_cache, tmp_input2);
    at::functionalization::impl::commit_update(key_cache);
    at::functionalization::impl::commit_update(value_cache);
    at::functionalization::impl::sync(key_cache);
    at::functionalization::impl::sync(value_cache);
    return ::std::tuple<at::Tensor, at::Tensor>(key_cache, value_cache);
}
} // namespace functionalization

namespace {
// regist Functionalize
TORCH_LIBRARY_IMPL(npu, Functionalize, m) {
    m.impl("npu_scatter_pa_kv_cache.value", TORCH_FN(functionalization::npu_scatter_pa_kv_cache));
}

}  // namespace

} // namespace at_npu