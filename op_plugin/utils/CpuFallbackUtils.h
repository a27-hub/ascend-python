#pragma once

#include <string>

#include "torch_npu/csrc/framework/utils/CpuFallbackUtils.h"

namespace op_plugin {
namespace utils {

inline void CheckCpuFallbackAllowed(
    const std::string& op_name,
    const std::string& reason) {
  at_npu::native::CheckCpuFallbackAllowed(
      op_name,
      at_npu::native::CpuFallbackKind::ExplicitKernel,
      reason);
}

} // namespace utils
} // namespace op_plugin

#define OP_PLUGIN_CPU_FALLBACK_CHECK(op_name, reason) \
  ::op_plugin::utils::CheckCpuFallbackAllowed((op_name), (reason))