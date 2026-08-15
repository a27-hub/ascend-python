#pragma once

#include <string>

#include "torch_npu/csrc/framework/utils/CpuFallbackUtils.h"

namespace op_plugin {
namespace utils {

inline void CheckCpuFallbackAllowed(
    const std::string& op_name,
    const std::string& interface_name,
    const std::string& reason) {
  at_npu::native::CheckCpuFallbackAllowed(
      op_name,
      at_npu::native::CpuFallbackKind::ExplicitKernel,
      reason);
}

} // namespace utils
} // namespace op_plugin

#define OP_PLUGIN_CPU_FALLBACK_CHECK(op_name, interface_name, reason)                     \
  do {                                                                                    \
    ::op_plugin::utils::CheckCpuFallbackAllowed((op_name), (interface_name), (reason));   \
    TORCH_NPU_WARN_ONCE(                                                                  \
        "CPU fallback is enabled by TORCH_NPU_CPU_FALLBACK (effective value: 1).\n"       \
        "Since interface '", (interface_name), "' does not support operator '",            \
        (op_name), "'\nfor the current execution condition (", (reason),                   \
        "), the operator will fall back to CPU.");                                        \
  } while (0)