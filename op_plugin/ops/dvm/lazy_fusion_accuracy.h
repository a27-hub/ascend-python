// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DVM_LAZY_FUSION_ACCURACY_H
#define DVM_LAZY_FUSION_ACCURACY_H

#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>

namespace lazy_fusion {

// Tolerance for one dtype. Mirrors the Python-side `acc_comp_tol` entries in
// `torch_npu/_inductor/config.py`.
struct AccuracyTol {
  double rtol{0.0};
  double atol{0.0};
};

// Accuracy comparison configuration. Default values are aligned with
// `torch_npu/_inductor/config.py` (`rtol_f32/f16/bf16/default`, `atol_default`).
// The custom rtol/atol can be overridden via the same env var as the graph-mode
// path: `INDUCTOR_ASCEND_CHECK_ACCURACY_RTOL_ATOL="rtol=1e-6,atol=1e-7"`.
struct AccuracyConfig {
  AccuracyTol f32{1.3e-6, 1e-5};
  AccuracyTol f16{1e-3, 1e-5};
  AccuracyTol bf16{1.6e-2, 1e-5};
  AccuracyTol default_tol{1.3e-6, 1e-5};

  // Parse the env var once and cache. Format: "rtol=1e-6,atol=1e-7".
  // When a value is set, it overrides every dtype entry (same behaviour as the
  // Python `parse_rtol_atol` in config.py).
  static const AccuracyConfig &Instance() {
    static const AccuracyConfig cfg = ParseFromEnv();
    return cfg;
  }

  const AccuracyTol &Get(at::ScalarType dtype) const {
    switch (dtype) {
      case at::ScalarType::Float:
        return f32;
      case at::ScalarType::Half:
        return f16;
      case at::ScalarType::BFloat16:
        return bf16;
      default:
        return default_tol;
    }
  }

 private:
  static AccuracyConfig ParseFromEnv() {
    AccuracyConfig cfg;
    const char *env = std::getenv("INDUCTOR_ASCEND_CHECK_ACCURACY_RTOL_ATOL");
    if (env == nullptr) {
      return cfg;
    }
    std::string s(env);
    double rtol = -1.0;
    double atol = -1.0;
    // Split by ',' and parse each "key=value" pair.
    size_t start = 0;
    while (start <= s.size()) {
      size_t pos = s.find(',', start);
      std::string token = (pos == std::string::npos) ? s.substr(start) : s.substr(start, pos - start);
      // trim
      auto first = token.find_first_not_of(" \t");
      auto last = token.find_last_not_of(" \t");
      if (first != std::string::npos && last != std::string::npos) {
        token = token.substr(first, last - first + 1);
        auto eq = token.find('=');
        if (eq != std::string::npos) {
          std::string key = token.substr(0, eq);
          std::string val = token.substr(eq + 1);
          // trim key/val
          auto kf = key.find_first_not_of(" \t");
          auto kl = key.find_last_not_of(" \t");
          auto vf = val.find_first_not_of(" \t");
          auto vl = val.find_last_not_of(" \t");
          if (kf != std::string::npos && kl != std::string::npos) {
            key = key.substr(kf, kl - kf + 1);
          }
          if (vf != std::string::npos && vl != std::string::npos) {
            val = val.substr(vf, vl - vf + 1);
          }
          try {
            double v = std::stod(val);
            if (key == "rtol" || key == "RTOL" || key == "Rtol") {
              rtol = v;
            } else if (key == "atol" || key == "ATOL" || key == "Atol") {
              atol = v;
            }
          } catch (...) {
            // ignore malformed value
          }
        }
      }
      if (pos == std::string::npos) {
        break;
      }
      start = pos + 1;
    }
    // rtol/atol stay at the -1.0 "unset" sentinel unless the env supplied them;
    // >= 0 lets a user request an exact-match (rtol=0,atol=0) comparison.
    if (rtol >= 0.0) {
      cfg.f32.rtol = cfg.f16.rtol = cfg.bf16.rtol = cfg.default_tol.rtol = rtol;
    }
    if (atol >= 0.0) {
      cfg.f32.atol = cfg.f16.atol = cfg.bf16.atol = cfg.default_tol.atol = atol;
    }
    return cfg;
  }
};

// Clone a tensor for accuracy comparison. bfloat16 is promoted to float32 to
// match the Python-side `clone_for_accuracy` in npu_compare.py.
// Non-tensor arguments are returned unchanged (handled by caller).
inline at::Tensor CloneForAccuracy(const at::Tensor &t) {
  if (!t.defined()) {
    return t;
  }
  auto cloned = t.clone();
  if (cloned.scalar_type() == at::ScalarType::BFloat16) {
    return cloned.to(at::kFloat);
  }
  return cloned;
}

// Resolves an op's tensor argument to the tensor that the op_api reference
// implementation should consume, during the eager accuracy replay.
//
// A single per-fusion `data_ptr -> reference tensor` map threads the references:
//   - An intermediate produced by an earlier fused op was registered in the map
//     under its output data_ptr, so a later op reading that tensor resolves to
//     the earlier op's *reference* result (not the still-unmaterialized DVM
//     fused buffer).
//   - An external input not yet in the map is cloned on first sight (bf16 ->
//     fp32, matching the DVM compute path) and memoized, so repeated reads share
//     one snapshot and the comparison is isolated from in-place mutation.
// Resolution runs eagerly inside the op entry (see LF_ACC), so every argument --
// including TensorList / ScalarArray / string_view views -- is still alive; no
// argument needs to be deep-copied for deferred replay.
class Resolver {
 public:
  explicit Resolver(std::unordered_map<void *, at::Tensor> &ref_map) : ref_map_(ref_map) {}

  at::Tensor operator()(const at::Tensor &t) const {
    if (!t.defined()) {
      return t;
    }
    void *ptr = t.data_ptr();
    auto it = ref_map_.find(ptr);
    if (it != ref_map_.end()) {
      return it->second;
    }
    at::Tensor cloned = CloneForAccuracy(t);
    ref_map_.emplace(ptr, cloned);
    return cloned;
  }

  c10::optional<at::Tensor> operator()(const c10::optional<at::Tensor> &t) const {
    if (!t.has_value()) {
      return c10::nullopt;
    }
    return (*this)(t.value());
  }

  std::vector<at::Tensor> operator()(at::TensorList tensors) const {
    std::vector<at::Tensor> resolved;
    resolved.reserve(tensors.size());
    for (const auto &t : tensors) {
      resolved.push_back((*this)(t));
    }
    return resolved;
  }

 private:
  std::unordered_map<void *, at::Tensor> &ref_map_;
};

// Report a mismatch for one output. Output format is aligned with the Python
// `_report_mismatch` in npu_compare.py so logs are consistent across modes.
inline void ReportMismatch(int64_t idx, const at::Tensor &actual, const at::Tensor &expected,
                           const at::Tensor &matches, double rtol, double atol, const std::string &kernel_name,
                           const std::string &ops_str = "") {
  at::Tensor abs_diff;
  try {
    abs_diff = at::abs(actual - expected);
  } catch (...) {
    abs_diff = at::abs(actual.to(at::kFloat) - expected.to(at::kFloat));
  }
  at::Tensor expected_abs = at::abs(expected);
  if (!expected_abs.is_floating_point() && !expected_abs.is_complex()) {
    expected_abs = expected_abs.to(at::kFloat);
  }
  at::Tensor rel_diff = abs_diff / at::clamp(expected_abs, /*min=*/1e-20);
  rel_diff.masked_fill_(matches, 0);
  int64_t number_of_elements = matches.numel();
  int64_t total_mismatches = number_of_elements - matches.sum().item<int64_t>();
  std::cerr << "[ERROR] DVM accuracy mismatch! "
            << "Kernel: " << kernel_name << ", Output idx: " << idx << ", "
            << "Mismatched: " << total_mismatches << "/" << number_of_elements << " ("
            << (number_of_elements == 0 ? 0.0
                                        : static_cast<double>(total_mismatches) / number_of_elements * 100.0)
            << "%), "
            << "Greatest Rel Diff: " << rel_diff.max().item<double>() << ", "
            << "Greatest Abs Diff: " << abs_diff.max().item<double>() << ", "
            << "rtol: " << rtol << ", atol: " << atol;
  if (!ops_str.empty()) {
    std::cout << ", ops: [" << ops_str << "]";
  }
  std::cout << std::endl;
}

// Compare actual vs expected outputs element-wise. Returns true if all outputs
// match within tolerance. Aligned with `compare_outputs` in npu_compare.py.
// `actual_outputs` are the DVM fused results; `expected_outputs` are the
// op_api reference results.
inline bool CompareOutputs(const std::vector<at::Tensor> &actual_outputs,
                           const std::vector<at::Tensor> &expected_outputs, const std::string &kernel_name,
                           const std::string &ops_str = "") {
  const auto &tol_cfg = AccuracyConfig::Instance();
  bool all_passed = true;
  size_t n = std::min(actual_outputs.size(), expected_outputs.size());
  for (size_t i = 0; i < n; ++i) {
    const auto &actual = actual_outputs[i];
    const auto &expected = expected_outputs[i];
    if (!actual.defined() || !expected.defined()) {
      continue;
    }
    if (!actual.is_floating_point() && !expected.is_floating_point()) {
      // Non-floating point: require exact equality.
      if (!at::equal(actual, expected)) {
        std::cerr << "[ERROR] DVM accuracy mismatch! "
                  << "Kernel: " << kernel_name << ", Output idx: " << i << ", "
                  << "Mismatched (exact equality required for non-floating dtype)";
        if (!ops_str.empty()) {
          std::cerr << ", ops: [" << ops_str << "]";
        }
        std::cerr << std::endl;
        all_passed = false;
      }
      continue;
    }
    at::Tensor expected_cast = expected;
    if (actual.scalar_type() != expected.scalar_type()) {
      expected_cast = expected.to(actual.scalar_type());
    }
    const auto &tol = tol_cfg.Get(actual.scalar_type());
    at::Tensor matches = at::isclose(actual, expected_cast, /*rtol=*/tol.rtol, /*atol=*/tol.atol,
                                     /*equal_nan=*/true);
    if (!matches.all().item<bool>()) {
      ReportMismatch(static_cast<int64_t>(i), actual, expected_cast, matches, tol.rtol, tol.atol, kernel_name,
                     ops_str);
      all_passed = false;
    }
  }
  return all_passed;
}

}  // namespace lazy_fusion

#endif  // DVM_LAZY_FUSION_ACCURACY_H
