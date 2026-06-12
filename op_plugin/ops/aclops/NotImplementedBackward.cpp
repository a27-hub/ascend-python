#include <torch/library.h>
#include <torch/csrc/autograd/autograd_not_implemented_fallback.h>

// These npu_ operators do NOT have a backward (gradient) implementation.
//
// Goal: when the user backpropagates through them, print a WARNING that the
// operator has no backward support, WITHOUT interrupting training.
//
// We register torch::autograd::basicAutogradNotImplementedFallback() on the
// AutogradPrivateUse1 key for each of them. This fallback:
//   * runs the FORWARD normally by redispatching to the backend kernel, so the
//     op still returns its correct outputs. (A hand-written no-op boxed kernel
//     would instead leave the input arguments on the stack and break forward
//     unpacking -- this is what caused the previous
//     "too many values to unpack" errors.)
//   * installs a WarnNotImplemented grad_fn on outputs that require grad. When
//     .backward() reaches it, it prints a warning and returns undefined (None)
//     gradients. Training CONTINUES, it is NOT interrupted.
//   * if no input requires grad (inference / forward-only), it is a plain
//     redispatch with no warning and no overhead.
//
// NOTE: the default global autograd fallback mode is already Warn (see
// torch/csrc/autograd/autograd_not_implemented_fallback.cpp), so we do not call
// setAutogradFallbackMode() -- that switch is process-global and would affect
// every other custom operator. The Error mode is also NYI in PyTorch.
TORCH_LIBRARY_IMPL(npu, AutogradPrivateUse1, m)
{
    m.impl("npu_add_rms_norm", torch::autograd::basicAutogradNotImplementedFallback());
    m.impl("npu_apply_rotary_pos_emb", torch::autograd::basicAutogradNotImplementedFallback());
    m.impl("npu_interleave_rope", torch::autograd::basicAutogradNotImplementedFallback());
    m.impl("npu_moe_gating_top_k_softmax", torch::autograd::basicAutogradNotImplementedFallback());
}